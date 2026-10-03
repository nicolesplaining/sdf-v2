"""LoRA synthetic document finetuning with a pretraining-style next-token loss.

Documents flagged with doctag=True get a "<DOCTAG>" prefix that is masked from the loss
(Slocum et al. 2025). Train on --model; to graft, train on the base checkpoint and apply the
saved adapter to the instruct model at eval time (Nutter et al. 2026).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

DOCTAG = "<DOCTAG>"


def tokenize(rows, tok, max_len):
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    tag_ids = tok(DOCTAG, add_special_tokens=False)["input_ids"]
    out = []
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"] + [eot]
        if r.get("doctag"):
            ids, labels = tag_ids + ids, [-100] * len(tag_ids) + ids
        else:
            labels = list(ids)
        out.append((ids[:max_len], labels[:max_len]))
    return out


def batches(examples, bs, seed, pad_id):
    """Length-grouped random batches: shuffle, sort within chunks of 50 batches, split."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    chunk = bs * 50
    groups = []
    for i in range(0, len(idx), chunk):
        c = sorted(idx[i : i + chunk], key=lambda j: len(examples[j][0]))
        groups += [c[k : k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(groups)
    for g in groups:
        L = max(len(examples[j][0]) for j in g)
        ids = torch.full((len(g), L), pad_id)
        labels = torch.full((len(g), L), -100)
        mask = torch.zeros((len(g), L), dtype=torch.long)
        for b, j in enumerate(g):
            x, y = examples[j]
            ids[b, : len(x)] = torch.tensor(x)
            labels[b, : len(y)] = torch.tensor(y)
            mask[b, : len(x)] = 1
        yield ids, labels, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16, help="effective batch size (docs per optimizer step)")
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None, help="truncate the corpus (smoke tests)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)][: args.max_docs]
    examples = tokenize(rows, tok, args.max_len)
    n_tokens = sum(sum(1 for y in ex[1] if y != -100) for ex in examples)
    print(f"{len(examples)} docs, {n_tokens / 1e6:.1f}M trained tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    accum = max(1, args.bs // args.micro_bs)
    steps_per_epoch = math.ceil(len(examples) / (args.micro_bs * accum))
    total = steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_frac * total), total)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        micro = list(batches(examples, args.micro_bs, args.seed + epoch, pad_id))
        for i in range(0, len(micro), accum):
            group = micro[i : i + accum]
            n_tok = sum((y != -100).sum().item() for _, y, _ in group)
            for ids, labels, mask in group:
                ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
                fwd = model(input_ids=ids, attention_mask=mask)
                logits = fwd.logits[:, :-1].float()
                tgt = labels[:, 1:]
                loss_sum = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum")
                (loss_sum / n_tok).backward()
                del fwd, logits
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == total:
                el = time.time() - t0
                rec = {"step": step, "loss": round(loss_sum.item() / max(1, (tgt != -100).sum().item()), 4), "lr": sched.get_last_lr()[0], "elapsed_min": round(el / 60, 1)}
                log.append(rec)
                print(json.dumps(rec), f"eta_min={el / step * (total - step) / 60:.1f}", flush=True)

    model.save_pretrained(out / "adapter")
    json.dump({"args": vars(args), "n_docs": len(examples), "n_tokens": n_tokens, "log": log},
              open(out / "train_log.json", "w"), indent=1)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
