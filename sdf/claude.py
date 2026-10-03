"""Thin wrapper around the Anthropic Message Batches API with cost logging."""

import json
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# $ per 1M tokens (input, output); batches are billed at 50%.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5-5": (2.0, 10.0)}


def load_env(path: str = ".env"):
    if Path(path).exists():
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def client() -> anthropic.Anthropic:
    load_env()
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)


def spent_since_topup(log_path: str) -> tuple[float, float | None]:
    """(spent, limit) counted from the last {"tag": "topup", "limit": X} line in the cost log."""
    spent, limit = 0.0, None
    if Path(log_path).exists():
        for line in open(log_path):
            r = json.loads(line)
            if r.get("tag") == "topup":
                spent, limit = 0.0, r["limit"]
            else:
                spent += r.get("cost_usd", 0.0)
    return spent, limit


def run_batch(requests: list[dict], model: str, max_tokens: int = 16000, log_path: str = "runs/api_costs.jsonl",
              tag: str = "", poll_s: int = 20, extra_params: dict | None = None,
              reuse_ids: list[str] | None = None, est_output_tokens: int = 1000) -> dict[str, str]:
    """requests: [{"custom_id": str, "messages": [...]}]. Returns {custom_id: text} for successes.

    reuse_ids: ids of already-finished batches for these same requests; results are read instead of resubmitting.
    """
    c = client()
    reqs = [
        Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(
            model=model, max_tokens=max_tokens, messages=r["messages"], **(extra_params or {})))
        for r in requests
    ]
    n_new = len(requests) - 50_000 * len(reuse_ids or [])
    if n_new > 0:  # budget guard: rough estimate (chars/3.5 input, est_output_tokens each), batch price
        chars = sum(len(m["content"]) for r in requests[-n_new:] for m in r["messages"])
        pin, pout = PRICES[model]
        est = 0.5 * (chars / 3.5 * pin + n_new * est_output_tokens * pout) / 1e6
        spent, limit = spent_since_topup(log_path)
        if limit is not None and spent + est > limit:
            raise RuntimeError(f"[{tag}] budget guard: est ${est:.2f} + spent ${spent:.2f} > limit ${limit:.2f}")
        print(f"[{tag}] estimated cost ${est:.2f} (spent ${spent:.2f} of {limit})", flush=True)
    out: dict[str, str] = {}
    usage_in = usage_out = 0
    chunks = [reqs[i : i + 50_000] for i in range(0, len(reqs), 50_000)]
    for k, chunk in enumerate(chunks):
        reused = bool(reuse_ids and k < len(reuse_ids))
        if reused:
            batch = c.messages.batches.retrieve(reuse_ids[k])
            print(f"[{tag}] reusing batch {batch.id}", flush=True)
        else:
            batch = c.messages.batches.create(requests=chunk)
            print(f"[{tag}] batch {batch.id}: {len(chunk)} requests", flush=True)
        while True:
            b = c.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            rc = b.request_counts
            print(f"[{tag}] processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored}", flush=True)
            time.sleep(poll_s)
        for res in c.messages.batches.results(batch.id):
            if res.result.type != "succeeded":
                continue
            msg = res.result.message
            if not reused:  # already logged when the batch first ran
                usage_in += msg.usage.input_tokens
                usage_out += msg.usage.output_tokens
            if msg.stop_reason == "refusal":
                continue
            out[res.custom_id] = "".join(bl.text for bl in msg.content if bl.type == "text")
    pin, pout = PRICES.get(model, (0, 0))
    cost = 0.5 * (usage_in * pin + usage_out * pout) / 1e6
    rec = {"tag": tag, "model": model, "n_requests": len(reqs), "n_ok": len(out),
           "input_tokens": usage_in, "output_tokens": usage_out, "cost_usd": round(cost, 2), "time": time.ctime()}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{tag}] done: {len(out)}/{len(reqs)} ok, ${cost:.2f}", flush=True)
    return out
