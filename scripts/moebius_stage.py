#!/usr/bin/env python3 -u
"""
MOEBIUS STAGE — full-live cross-machine orchestration of the P39/MS17
standing-replica claim, over TWO REAL MACHINES connected by a REAL NETWORK.

  Organism A lives on the Mac (this machine) as ONE continuous
  `portable_organism.py --run` subprocess, started once and never paused,
  never killed, for the entire staging — the "never 100%, always live"
  source. Organism B is a SEPARATE machine (beast, x86, over SSH) that
  bootstraps from A's snapshots via `--resume` and chases A's live stream
  position: catch up, briefly run in lockstep, measure how far behind it
  is, then catch up again on a fresh snapshot. Repeated for k cycles.

This script is a PURE ORCHESTRATOR over src/portable_organism.py's public
CLI (--run / --resume / --chunks / --ckpt-every / --d-model / --out-dir /
--result-json / --eval-every). It does not import portable_organism's
internals and does not reimplement any training/streaming logic — every
number in the output JSON is either (a) read directly from a
result-JSON/checkpoint that portable_organism.py itself wrote, or (b) a
wall-clock/chunk-count measurement of the orchestration itself (transfer
time, catch-up time, snapshot polling).

Ablauf pro Zyklus i (0-indexed):
  (i)   A already runs continuously in the background (started once, before
        cycle 0, with --chunks set high enough to outlive every cycle).
        Wait for A's snapshot checkpoint (out_dir/ckpt.pt, overwritten
        atomically by --run every --ckpt-every chunks) to advance past the
        chunk index this cycle targets.
  (ii)  scp the snapshot to beast (Messung: transfer_s), then ssh beast to
        run B's catch-up: --resume <snapshot> --chunks <T> (T = A's chunk
        count at snapshot time minus B's current chunk count) — deterministic
        stream replay to close the gap (Messung: catchup_s, catchup_chunks).
  (iii) Lockstep window: B streams M more chunks (--resume on its own new
        ckpt.pt). Compare heldout from B's result-JSON against a fresh
        readout of A's live snapshot (both are point measurements — cross-
        ISA parity is 6-decimal per P38a, NOT bit-identical, beast is x86).
  (iv)  Sync debt: read A's live chunk count again (fresh snapshot readout)
        and subtract B's chunk count after lockstep. Registered expectation
        (P39d): debt stays bounded, <= one sync window (the cadence between
        snapshots), and does not grow cycle over cycle.

A is NEVER stopped, NEVER paused, NEVER killed by this script — not even
between cycles. All ssh calls are stdin-detached (< /dev/null equivalent,
via stdin=DEVNULL) so the orchestrator session never hangs on them. A
failed transfer or catch-up aborts ONLY that cycle (marked "failed" in the
output JSON) and the loop proceeds to the next cycle; it never touches A.

Usage:
  python3 scripts/moebius_stage.py --dry-run
  python3 scripts/moebius_stage.py --cycles 3 --sync-window 40 --lockstep 20 \
      --transfer-budget 40 --d-model 128

beast side: nothing outside /root/o1lab is touched. B always runs there via
`nice -n 19` (beast carries production services). All beast paths are
absolute under /root/o1lab so this never collides with anything else there.
"""
import os
import sys
import json
import time
import shlex
import shutil
import argparse
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTABLE = os.path.join(REPO_ROOT, "src", "portable_organism.py")

SSH_HOST = "beast"
BEAST_ROOT = "/root/o1lab"
BEAST_PY = "/usr/bin/python3"

# P39d scoring tolerances (registered expectations, per the task brief).
PARITY_TOL = 1e-5          # heldout diff, 6-decimal cross-ISA parity (P38a)


# ═══════════════════════════════════════════════════════════════════════════
#  Small process helpers
# ═══════════════════════════════════════════════════════════════════════════
def _run(cmd, dry_run, label, **kw):
    """Runs (or, under --dry-run, just prints) a local subprocess command.
    Returns a CompletedProcess-like object; under dry-run, returncode=0 and
    empty stdout/stderr so callers can proceed through the dry-run path
    without special-casing every call site."""
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[dry-run] {label}: {printable}" if dry_run else f"[run] {label}: {printable}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _ssh_cmd(remote_cmd):
    """ssh with stdin detached (equivalent of `< /dev/null`) so a hung remote
    command can never block this orchestrator's own stdin/session."""
    return ["ssh", SSH_HOST, remote_cmd]


def _run_ssh(remote_cmd, dry_run, label):
    cmd = _ssh_cmd(remote_cmd)
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[dry-run] {label}: {printable}  (stdin=/dev/null)" if dry_run
          else f"[run] {label}: {printable}  (stdin=/dev/null)", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    with open(os.devnull, "r") as devnull:
        return subprocess.run(cmd, capture_output=True, text=True, stdin=devnull)


def _scp_to_beast(local_path, remote_path, dry_run, label):
    cmd = ["scp", local_path, f"{SSH_HOST}:{remote_path}"]
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[dry-run] {label}: {printable}" if dry_run else f"[run] {label}: {printable}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    with open(os.devnull, "r") as devnull:
        return subprocess.run(cmd, capture_output=True, text=True, stdin=devnull)


def _scp_from_beast(remote_path, local_path, dry_run, label):
    cmd = ["scp", f"{SSH_HOST}:{remote_path}", local_path]
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[dry-run] {label}: {printable}" if dry_run else f"[run] {label}: {printable}", flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    with open(os.devnull, "r") as devnull:
        return subprocess.run(cmd, capture_output=True, text=True, stdin=devnull)


# ═══════════════════════════════════════════════════════════════════════════
#  Reading a portable_organism.py snapshot's metadata WITHOUT reimplementing
#  any training/streaming logic — this is the same torch.load() the
#  official --resume path uses internally; we just peek at organism.n_chunks
#  and wall_s to know where A currently is, without stepping the model.
# ═══════════════════════════════════════════════════════════════════════════
def _read_snapshot_meta(path, dry_run):
    """Returns {"n_chunks": int, "wall_s": float, "docs": int} read straight
    off a snapshot file's checkpoint dict, or None under --dry-run / on any
    read failure (e.g. file mid-write — atomic os.replace() in
    save_snapshot() means this should not happen against the FINAL path, but
    we guard anyway since this runs against a live, still-writing A)."""
    if dry_run:
        return {"n_chunks": None, "wall_s": None, "docs": None, "dry_run": True}
    try:
        import torch
        ck = torch.load(path, weights_only=False, map_location="cpu")
        return {
            "n_chunks": ck["organism"]["n_chunks"],
            "wall_s": ck["wall_s"],
            "docs": ck["stream_docs"],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _wait_for_a_snapshot_at_least(ckpt_path, min_chunks, dry_run, timeout=1800.0, poll=1.0):
    """Polls A's checkpoint file (overwritten atomically every --ckpt-every
    chunks by --run) until its n_chunks >= min_chunks. A keeps streaming the
    entire time this function polls — polling reads the FILE, it never
    touches A's process."""
    if dry_run:
        print(f"[dry-run] would poll {ckpt_path} until n_chunks >= {min_chunks} "
              f"(timeout {timeout}s, poll {poll}s)", flush=True)
        return {"n_chunks": min_chunks, "wall_s": None, "docs": None, "dry_run": True}, 0.0
    t0 = time.time()
    last_seen = None
    while True:
        if os.path.exists(ckpt_path):
            meta = _read_snapshot_meta(ckpt_path, dry_run=False)
            if "error" not in meta:
                last_seen = meta
                if meta["n_chunks"] >= min_chunks:
                    return meta, time.time() - t0
        if time.time() - t0 > timeout:
            raise RuntimeError(
                f"timed out after {timeout}s waiting for {ckpt_path} to reach "
                f"n_chunks>={min_chunks} (last seen: {last_seen})")
        time.sleep(poll)


# ═══════════════════════════════════════════════════════════════════════════
#  A: the continuous Mac-side source. Started ONCE, before cycle 0, with a
#  --chunks budget large enough to outlive the whole staged run (all cycles
#  plus slack). Never touched again by this script.
# ═══════════════════════════════════════════════════════════════════════════
def launch_source_a(args, out_dir):
    os.makedirs(out_dir, exist_ok=True) if not args.dry_run else None
    cmd = [
        sys.executable, "-u", PORTABLE,
        "--run", "--name", "moebius_A",
        "--chunks", str(args.a_total_chunks),
        "--ckpt-every", str(args.sync_window),
        "--d-model", str(args.d_model),
        "--out-dir", out_dir,
    ]
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"[dry-run] launch A (background, never stopped): {printable}" if args.dry_run
          else f"[run] launch A (background, never stopped): {printable}", flush=True)
    if args.dry_run:
        return None
    log_path = os.path.join(out_dir, "A_stdout.log")
    log_f = open(log_path, "a")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    print(f"[moebius] A launched: pid={proc.pid}, log={log_path} — "
          f"this process is NEVER paused/killed by this script", flush=True)
    return proc


# ═══════════════════════════════════════════════════════════════════════════
#  B: beast-side catch-up / lockstep, via ssh + the official --resume CLI.
# ═══════════════════════════════════════════════════════════════════════════
def beast_catchup(args, snapshot_remote_path, chunks, cycle_dir_remote, tag, dry_run):
    """ssh beast: nice -n 19 python3 src/portable_organism.py --resume ...
    --chunks <chunks> --out-dir <cycle_dir_remote> --result-json <path>
    --d-model <d>. Returns (result_dict_or_None, wall_s, ok, remote_result_path)."""
    remote_result_json = f"{cycle_dir_remote}/{tag}_result.json"
    remote_portable = f"{BEAST_ROOT}/src/portable_organism.py"
    remote_cmd = (
        f"cd {shlex.quote(BEAST_ROOT)} && "
        f"mkdir -p {shlex.quote(cycle_dir_remote)} && "
        f"nice -n 19 {shlex.quote(BEAST_PY)} -u {shlex.quote(remote_portable)} "
        f"--resume {shlex.quote(snapshot_remote_path)} "
        f"--chunks {int(chunks)} "
        f"--out-dir {shlex.quote(cycle_dir_remote)} "
        f"--result-json {shlex.quote(remote_result_json)} "
        f"--d-model {int(args.d_model)}"
    )
    t0 = time.time()
    proc = _run_ssh(remote_cmd, dry_run, f"B {tag} (chunks={chunks})")
    wall_s = time.time() - t0
    if dry_run:
        return {"dry_run": True}, wall_s, True, remote_result_json
    if proc.returncode != 0:
        print(proc.stdout[-3000:], flush=True)
        print(proc.stderr[-3000:], flush=True)
        return None, wall_s, False, remote_result_json

    # pull the result JSON back so we can read heldout/chunk_log locally
    local_result_json = os.path.join(args.local_out_dir, "cycles", f"cycle_result_{tag}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(local_result_json), exist_ok=True)
    scp_proc = _scp_from_beast(remote_result_json, local_result_json, dry_run, f"pull {tag} result")
    if scp_proc.returncode != 0:
        print(scp_proc.stdout[-1000:], flush=True)
        print(scp_proc.stderr[-1000:], flush=True)
        return None, wall_s, False, remote_result_json
    try:
        with open(local_result_json) as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[moebius] failed to parse pulled result json: {e}", flush=True)
        return None, wall_s, False, remote_result_json
    return result, wall_s, True, remote_result_json


# ═══════════════════════════════════════════════════════════════════════════
#  One cycle: snapshot -> transfer -> catch-up -> lockstep -> parity -> debt
# ═══════════════════════════════════════════════════════════════════════════
def run_cycle(args, cycle_idx, a_ckpt_path, b_chunk_before, b_last_ckpt_remote):
    """b_chunk_before: B's chunk count BEFORE this cycle's catch-up (0 for
    cycle 0, since B does not exist yet). b_last_ckpt_remote: remote path of
    B's own previous checkpoint (None for cycle 0 — cycle 0 bootstraps
    straight from A's snapshot instead of from a prior B checkpoint)."""
    dry = args.dry_run
    cycle_tag = f"cycle{cycle_idx}"
    cycle_dir_remote = f"{BEAST_ROOT}/moebius/{cycle_tag}"
    cycle_dir_local = os.path.join(args.local_out_dir, "cycles", cycle_tag)
    if not dry:
        os.makedirs(cycle_dir_local, exist_ok=True)

    record = {"cycle": cycle_idx, "status": "started"}

    # ── (i) wait for A to have advanced at least one fresh sync window
    #    past wherever B currently is (or past 0 for cycle 0). ──────────────
    target_min_chunks = b_chunk_before + args.sync_window
    print(f"\n[moebius] === cycle {cycle_idx}: waiting for A >= chunk {target_min_chunks} ===", flush=True)
    a_meta, wait_s = _wait_for_a_snapshot_at_least(a_ckpt_path, target_min_chunks, dry)
    record["a_snapshot_wait_s"] = round(wait_s, 3)
    record["a_chunks_at_snapshot"] = a_meta.get("n_chunks")

    # ── (ii) transfer snapshot to beast (Messung: transfer_s) ──────────────
    remote_snap = f"{cycle_dir_remote}/snap_from_A.pt"
    if not dry:
        _run_ssh(f"mkdir -p {shlex.quote(cycle_dir_remote)}", dry, "mkdir cycle dir on beast")
    t0 = time.time()
    scp_proc = _scp_to_beast(a_ckpt_path, remote_snap, dry, f"transfer A snapshot -> beast ({cycle_tag})")
    transfer_s = time.time() - t0
    record["transfer_s"] = round(transfer_s, 3)
    if not dry and scp_proc.returncode != 0:
        print(scp_proc.stdout[-1000:], flush=True)
        print(scp_proc.stderr[-1000:], flush=True)
        record["status"] = "failed"
        record["failure"] = "scp_transfer"
        return record, b_chunk_before, b_last_ckpt_remote

    # ── (iii) B catches up: chunks = A's snapshot chunk count - B's current
    #    chunk count (deterministic replay, per portable_organism.py's
    #    --resume contract). ──────────────────────────────────────────────────
    a_chunks_at_snap = a_meta.get("n_chunks")
    if dry:
        catchup_chunks = args.sync_window  # nominal, for a readable dry-run printout
    else:
        catchup_chunks = max(0, a_chunks_at_snap - b_chunk_before)
    record["catchup_chunks"] = catchup_chunks

    catchup_result, catchup_s, ok, remote_catchup_json = beast_catchup(
        args, remote_snap, catchup_chunks, cycle_dir_remote, "catchup", dry)
    record["catchup_s"] = round(catchup_s, 3)
    if not ok:
        record["status"] = "failed"
        record["failure"] = "catchup"
        return record, b_chunk_before, b_last_ckpt_remote

    b_ckpt_after_catchup_remote = f"{cycle_dir_remote}/ckpt.pt"  # --run/--resume always write out_dir/ckpt.pt
    b_chunks_after_catchup = catchup_result.get("n_chunks") if not dry else (b_chunk_before + catchup_chunks)
    record["b_chunks_after_catchup"] = b_chunks_after_catchup
    record["catchup_final_heldout"] = catchup_result.get("final_heldout") if not dry else None

    # ── (iii cont'd) lockstep window: B runs M more chunks from its own new
    #    checkpoint, in step with A's live stream. ──────────────────────────
    lockstep_result, lockstep_s, ok, remote_lockstep_json = beast_catchup(
        args, b_ckpt_after_catchup_remote, args.lockstep, cycle_dir_remote, "lockstep", dry)
    record["lockstep_s"] = round(lockstep_s, 3)
    if not ok:
        record["status"] = "failed"
        record["failure"] = "lockstep"
        return record, b_chunks_after_catchup if not dry else b_chunk_before, b_ckpt_after_catchup_remote

    b_chunks_after_lockstep = lockstep_result.get("n_chunks") if not dry else (b_chunks_after_catchup + args.lockstep)
    b_heldout_after_lockstep = lockstep_result.get("final_heldout") if not dry else None
    record["b_chunks_after_lockstep"] = b_chunks_after_lockstep
    record["lockstep_final_heldout"] = b_heldout_after_lockstep

    # ── parity: compare B's post-lockstep heldout against A's heldout AT THE
    #    SAME CHUNK COUNT. Parity is only a meaningful claim at matched chunk
    #    positions — the underlying stream is deterministic (C4Stream's
    #    doc-count resume), so equal chunk count means equal stream position,
    #    which is what the 6-decimal cross-ISA match (P38a) is actually a
    #    claim about. A comparison at UNEQUAL chunk counts isn't a parity
    #    check at all — it's just measuring how much A trained in the
    #    meantime, and would spuriously "fail" even on a perfect migration.
    #    By construction (--lockstep validated as a multiple of
    #    --sync-window, see build_argparser), B's post-lockstep chunk count
    #    always lands exactly on one of A's --eval-every readout points, so
    #    the matching entry in A's metrics.jsonl either already exists (A is
    #    always ahead of B) or appears within a few eval cycles — we poll
    #    briefly for it rather than falling back to a mismatched neighbor.
    # ─────────────────────────────────────────────────────────────────────────
    parity = {"note": "cross-ISA (beast=x86) — 6-decimal match expected per P38a, NOT bit-identical; "
                      "only defined at equal chunk counts (equal stream position)"}
    if dry:
        parity["measured"] = False
        parity["reason"] = "dry-run"
    else:
        a_heldout = _wait_for_a_heldout_at(args, b_chunks_after_lockstep)
        parity["chunk_count_compared"] = b_chunks_after_lockstep
        parity["a_heldout_used"] = a_heldout
        parity["b_heldout"] = b_heldout_after_lockstep
        if a_heldout is not None and b_heldout_after_lockstep is not None:
            delta = abs(a_heldout - b_heldout_after_lockstep)
            parity["heldout_abs_delta"] = round(delta, 8)
            parity["within_tolerance"] = bool(delta <= PARITY_TOL)
            parity["measured"] = True
        else:
            parity["measured"] = False
            parity["reason"] = "no A heldout reading at the matching chunk count within the poll timeout"
    record["lockstep_parity"] = parity

    # ── sync debt: how many chunks is A ahead of B once this cycle's
    #    lockstep is done? Registered expectation P39d: bounded, <= one sync
    #    window, not growing across cycles. ───────────────────────────────────
    if dry:
        # nominal placeholder: the dry-run model has A and B advancing in
        # lockstep with no simulated wall-clock gap, so debt reads as 0 here
        # — the real run computes this from live snapshot reads instead.
        record["sync_debt_chunks"] = 0
    else:
        a_meta_final = _read_snapshot_meta(a_ckpt_path, dry_run=False)
        a_chunks_final = a_meta_final.get("n_chunks")
        record["a_chunks_at_debt_readout"] = a_chunks_final
        if a_chunks_final is not None and b_chunks_after_lockstep is not None:
            record["sync_debt_chunks"] = max(0, a_chunks_final - b_chunks_after_lockstep)
        else:
            record["sync_debt_chunks"] = None

    record["status"] = "ok"
    return record, b_chunks_after_lockstep, b_ckpt_after_catchup_remote


def _read_a_heldout_exact(metrics_path, target_chunk):
    """Scans A's metrics.jsonl for an entry with n_chunks EXACTLY equal to
    target_chunk and returns its heldout value, or None if no such entry
    exists yet."""
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("n_chunks") == target_chunk:
                return rec.get("heldout")
    return None


def _wait_for_a_heldout_at(args, target_chunk, timeout=60.0, poll=1.0):
    """Parity is only defined at an EXACT matching chunk count (see the
    comment at the call site) — --lockstep is validated as a multiple of
    --sync-window, so target_chunk always coincides with one of A's
    --eval-every readout points. A is always further ahead than B by
    construction (B is the one catching up), so the entry should already be
    in metrics.jsonl by the time B finishes its lockstep; we poll briefly
    (not indefinitely — a short, bounded wait) to absorb the timing edge
    case where A's file write for that exact chunk hasn't landed yet."""
    metrics_path = os.path.join(args.local_out_dir, "source_A", "metrics.jsonl")
    t0 = time.time()
    while True:
        h = _read_a_heldout_exact(metrics_path, target_chunk)
        if h is not None:
            return h
        if time.time() - t0 > timeout:
            return None
        time.sleep(poll)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def build_argparser():
    ap = argparse.ArgumentParser(description="Moebius stage: full-live cross-machine "
                                             "standing-replica orchestration (P39d), Mac (A) <-> beast (B)")
    ap.add_argument("--d-model", type=int, default=128, help="organism width, forwarded to every subprocess")
    ap.add_argument("--cycles", type=int, default=3, help="number of snapshot+catchup+lockstep cycles")
    ap.add_argument("--sync-window", type=int, default=40,
                    help="A's --ckpt-every AND the target chunk gap per cycle (the 'sync window')")
    ap.add_argument("--lockstep", type=int, default=40,
                    help="chunks B runs in lockstep after each catch-up; MUST be a multiple of "
                         "--sync-window so B's post-lockstep chunk count always lands exactly on "
                         "one of A's --eval-every readout points (parity is only defined at equal "
                         "chunk counts — see run_cycle's parity comment)")
    ap.add_argument("--a-total-chunks", type=int, default=0,
                    help="A's total --chunks budget; 0 = auto (cycles * (sync_window+lockstep) * 4, generous slack)")
    ap.add_argument("--local-out-dir", default=os.path.join(REPO_ROOT, "results", "moebius_stage"),
                    help="local (Mac) output root for A's checkpoints/metrics and pulled-back B results")
    ap.add_argument("--result-json", default=os.path.join(REPO_ROOT, "results", "moebius_stage.json"),
                    help="final combined output JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command that WOULD run (including ssh/scp) without starting anything")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.sync_window <= 0:
        ap.error("--sync-window must be > 0")
    if args.lockstep % args.sync_window != 0:
        ap.error(f"--lockstep ({args.lockstep}) must be a multiple of --sync-window "
                 f"({args.sync_window}) — otherwise B's post-lockstep chunk count never lands on "
                 f"one of A's --eval-every readout points, and the parity check has nothing at an "
                 f"equal chunk count to compare against (see run_cycle's parity comment)")

    if args.a_total_chunks <= 0:
        args.a_total_chunks = args.cycles * (args.sync_window + args.lockstep) * 4

    if not args.dry_run:
        os.makedirs(args.local_out_dir, exist_ok=True)

    print(f"[moebius] config: d_model={args.d_model} cycles={args.cycles} "
          f"sync_window={args.sync_window} lockstep={args.lockstep} "
          f"a_total_chunks={args.a_total_chunks} dry_run={args.dry_run}", flush=True)

    a_out_dir = os.path.join(args.local_out_dir, "source_A")
    a_ckpt_path = os.path.join(a_out_dir, "ckpt.pt")

    # A is launched with --eval-every = sync_window so its metrics.jsonl
    # carries a heldout reading exactly once per sync window (chunk 0, sync_window,
    # 2*sync_window, ...) — --lockstep is validated as a multiple of --sync-window,
    # so B's post-lockstep chunk count always lands on one of these exact
    # readout points (see _wait_for_a_heldout_at / run_cycle's parity comment).
    a_cmd_preview = [
        sys.executable, "-u", PORTABLE, "--run", "--name", "moebius_A",
        "--chunks", str(args.a_total_chunks), "--ckpt-every", str(args.sync_window),
        "--eval-every", str(args.sync_window), "--d-model", str(args.d_model),
        "--out-dir", a_out_dir,
    ]
    if args.dry_run:
        printable = " ".join(shlex.quote(c) for c in a_cmd_preview)
        print(f"[dry-run] launch A (background, never stopped, --eval-every={args.sync_window}): {printable}",
              flush=True)
        a_proc = None
    else:
        os.makedirs(a_out_dir, exist_ok=True)
        log_path = os.path.join(a_out_dir, "A_stdout.log")
        log_f = open(log_path, "a")
        a_proc = subprocess.Popen(a_cmd_preview, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL)
        print(f"[moebius] A launched: pid={a_proc.pid}, log={log_path} — "
              f"NEVER paused/killed by this script, even on cycle failure", flush=True)

    # ensure the beast-side working dir exists (idempotent, scoped to
    # /root/o1lab/moebius only — nothing else on beast is touched)
    _run_ssh(f"mkdir -p {shlex.quote(BEAST_ROOT)}/moebius", args.dry_run, "ensure beast moebius dir")

    cycles_out = []
    b_chunk_count = 0
    b_last_ckpt_remote = None
    try:
        for i in range(args.cycles):
            record, b_chunk_count, b_last_ckpt_remote = run_cycle(
                args, i, a_ckpt_path, b_chunk_count, b_last_ckpt_remote)
            cycles_out.append(record)
            if record["status"] == "failed":
                print(f"[moebius] cycle {i} FAILED ({record.get('failure')}) — "
                      f"A keeps running, proceeding to next cycle", flush=True)
    finally:
        # A is intentionally NOT stopped here — it lives on after this script
        # exits, per the task's "A never pauses" constraint. We only report
        # its pid/log for whoever wants to check on or eventually stop it.
        pass

    # ── P39d scoring: debt bounded (<= sync window, every successful cycle),
    #    parity (heldout delta <= 1e-5, cross-ISA, every measured cycle).
    #    Under --dry-run nothing was actually measured, so both come back as
    #    None ("not evaluated") rather than False — a dry-run must never
    #    print PASS or FAIL, only show the commands it would have run. ───────
    ok_cycles = [c for c in cycles_out if c["status"] == "ok"]
    debts = [c["sync_debt_chunks"] for c in ok_cycles if c.get("sync_debt_chunks") is not None]
    parity_deltas = [c["lockstep_parity"]["heldout_abs_delta"] for c in ok_cycles
                     if c.get("lockstep_parity", {}).get("measured") and "heldout_abs_delta" in c["lockstep_parity"]]
    if args.dry_run:
        debt_bounded = None
        parity_ok = None
    else:
        debt_bounded = bool(debts) and all(d <= args.sync_window for d in debts)
        parity_ok = bool(parity_deltas) and all(d <= PARITY_TOL for d in parity_deltas)

    out = {
        "config": vars(args),
        "a_pid": a_proc.pid if a_proc else None,
        "a_ckpt_path": a_ckpt_path,
        "a_note": "A is a continuous subprocess left RUNNING after this script exits "
                 "(never paused/killed, per constraint) — kill it manually if/when desired.",
        "cycles": cycles_out,
        "p39d_scoring": {
            "debt_bounded": debt_bounded,
            "max_debt_chunks": max(debts) if debts else None,
            "sync_window_chunks": args.sync_window,
            "debts_by_cycle": debts,
            "parity": parity_ok,
            "parity_tolerance": PARITY_TOL,
            "parity_deltas_by_cycle": parity_deltas,
            "n_cycles_ok": len(ok_cycles),
            "n_cycles_total": len(cycles_out),
        },
    }
    if args.dry_run:
        verdict_tail = "DRY-RUN (nothing executed, nothing scored)"
    elif ok_cycles:
        verdict_tail = "PASS" if (debt_bounded and parity_ok) else "PARTIAL/FAIL"
    else:
        verdict_tail = "NO CYCLES COMPLETED"
    out["verdict"] = (
        f"MOEBIUS STAGE: {len(ok_cycles)}/{len(cycles_out)} cycles ok | "
        f"debt bounded (<= {args.sync_window} chunks, all cycles): {debt_bounded} "
        f"(max observed: {out['p39d_scoring']['max_debt_chunks']}) | "
        f"parity (<= {PARITY_TOL}, cross-ISA 6-decimal): {parity_ok} | "
        f"{verdict_tail}"
    )
    print(f"\n[moebius] {out['verdict']}", flush=True)

    if args.dry_run:
        print(f"\n[dry-run] would write combined result to {args.result_json}", flush=True)
        print(json.dumps(out, indent=2, default=str), flush=True)
        return

    d = os.path.dirname(args.result_json) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(tmp, args.result_json)
    print(f"[moebius] -> {args.result_json}", flush=True)


if __name__ == "__main__":
    main()
