# LIFETIME-RUN — never-restarting O(1) streaming trainer

A constant-memory streaming training process on the O(1) GSSM that collects
wallclock evidence from day one: flat RSS, a frozen-eval heartbeat that keeps
moving, an artifact whose value grows with every day of uptime.

It is a thin wrapper (`src/lifetime_run.py`) around the exact streaming path in
`src/streaming_train.py` — `StreamingNoPELM` + carried `Z.detach()` at chunk
boundaries + C4 streamed live from HuggingFace + a fixed WikiText-2 held-out slice.
No new architecture; the wrapper only adds the lifetime harness (unbounded runtime,
atomic checkpoints, status heartbeat, frozen eval, machine-safety pause).

---

## Where it runs

| | |
|---|---|
| **Server** | `intel` (89.167.47.205) |
| **Why intel** | Only fleet server with outbound DNS to `huggingface.co` (200 OK) — the run streams C4 live, so it needs HF reachability. `beast` has torch but **no DNS** (name resolution fails), so it cannot stream C4. `core` is 100% disk-full. |
| **Home dir** | `/root/o1_lifetime` |
| **venv** | `/root/o1_lifetime/.venv` (CPU torch + numpy datasets psutil) |
| **Threads** | `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`, `nice -n 19` (intel has 4 cores; JACSI/OJS is on beast, not intel) |
| **Start time (server clock)** | `2026-07-22T18:30:57Z` (UTC) |
| **PID** | `2071900` (python; launching shell was 2071899) |
| **First heartbeat** | +5 min: 1.82M tokens streamed, loss_ema 4.81, RSS 0.68 GB flat, 5,821 tok/s |

---

## Paths on the server

```
/root/o1_lifetime/
  src/            lifetime_run.py, streaming_train.py, length_extrap_v2.py, width_fix.py
  reference/      moebius_scan_transformer_selective.py, _sqrt.py, moebius_attention.py, ps_lifted_scan.py
  .venv/          CPU torch + deps
  lifetime.log    the full streaming log (line-buffered)
  results_lifetime/
    status.json          heartbeat: streamed_tokens, loss_ema, rss_gb, uptime, timestamp, pid
    eval_log.jsonl       frozen-eval heartbeat (one JSON line per 30-min eval), APPEND-only
    lifetime_ckpt.pt     atomic checkpoint (model + optimizer + streamed_tokens + step)
```

The HuggingFace dataset cache lives under `/root/.cache/huggingface` (WT-2 is
downloaded once; C4 is streamed and not cached to disk).

---

## Check it (heartbeat)

```bash
# machine-readable heartbeat
ssh intel cat ~/o1_lifetime/results_lifetime/status.json

# the frozen-eval curve (learning heartbeat — one line per 30 min)
ssh intel tail -5 ~/o1_lifetime/results_lifetime/eval_log.jsonl

# live log tail
ssh intel tail -20 ~/o1_lifetime/lifetime.log

# is it alive?
ssh intel 'ps -p <PID> -o pid,etime,rss,cmd 2>/dev/null || echo DEAD'
```

`status.json` fields: `streamed_tokens`, `step`, `loss_ema`, `rss_gb`,
`peak_rss_gb`, `uptime_h`, `tok_per_s`, `free_disk_gb`, `paused`, `note`, `pid`.

**Health = flat RSS + a `frozen_val_loss` in `eval_log.jsonl` that keeps drifting down.**

---

## Kill it

```bash
ssh intel kill <PID>
# if it does not stop within a few seconds:
ssh intel kill -9 <PID>
```

A clean `kill` lets the loop finish its step; the last atomic checkpoint on disk
is always complete (tmpfile + `os.replace`), so nothing is corrupted.

---

## Resume (the claim: it never restarts — but if it does, it continues)

The process is designed to run forever. If it is ever stopped (kill, reboot,
machine-safety pause that never clears), restart it with the **exact same command**
— on start it finds `lifetime_ckpt.pt`, loads model + optimizer + `streamed_tokens`,
and continues the stream from there. It never restarts training from zero.

```bash
ssh intel 'cd ~/o1_lifetime && \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  nohup nice -n 19 .venv/bin/python -u src/lifetime_run.py --hours 0 \
    >> ~/o1_lifetime/lifetime.log 2>&1 & echo "restarted PID $!"'
```

Look for `RESUMED from .../lifetime_ckpt.pt at N tok, step M` in the log.

---

## The exact start command (for reference / re-deploy)

```bash
ssh intel 'cd ~/o1_lifetime && \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  nohup nice -n 19 .venv/bin/python -u src/lifetime_run.py --hours 0 \
    > ~/o1_lifetime/lifetime.log 2>&1 & echo "PID $!"'
```

`--hours 0` = run forever. Any positive `--hours N` is a soft wallclock cap
(checkpoints and exits cleanly at N hours). Defaults: `d_model=128`, `batch=8`,
`chunk=64`, `lr=3e-3`, checkpoint every 10 min, frozen eval every 30 min on a
fixed 200k-token WT-2 slice, PAUSE if RSS > 4 GB or free disk < 10 GB, hard
watchdog SIGKILL at 6 GB.

---

## Machine-safety (why it will not hurt intel)

- **Threads capped** at 4 (`OMP/MKL_NUM_THREADS=4`) and `nice -n 19` — it yields
  to anything intel does in the foreground.
- **RSS pause** at 4 GB: the loop sleeps 30 s and re-checks instead of growing
  (our measured RSS is ~0.2–0.5 GB, so this is a wide margin).
- **Disk pause** at 10 GB free: if intel's disk drops below 10 GB free, the loop
  pauses rather than filling it. C4 is streamed, not stored; only checkpoints
  (~a few MB) and the two small heartbeat files are written.
- **Hard watchdog** SIGKILL at 6 GB (writes `status.json.WATCHDOG_KILL`) — the
  last line of defense above the soft pause.

## Re-deploy from scratch (if the server bundle is lost)

The minimal bundle is 9 files (`src/` 4 + `reference/` 4 + `requirements.txt`).
From the local repo `/Users/bhkmie/Documents/Forschung/O1_juli`:

```bash
scp -r src/lifetime_run.py src/streaming_train.py src/length_extrap_v2.py src/width_fix.py \
  intel:~/o1_lifetime/src/
scp -r reference/moebius_scan_transformer_selective.py reference/moebius_scan_transformer_sqrt.py \
  reference/moebius_attention.py reference/ps_lifted_scan.py intel:~/o1_lifetime/reference/
ssh intel 'cd ~/o1_lifetime && python3 -m venv .venv && . .venv/bin/activate && \
  pip install torch --index-url https://download.pytorch.org/whl/cpu && \
  pip install numpy datasets psutil'
```
