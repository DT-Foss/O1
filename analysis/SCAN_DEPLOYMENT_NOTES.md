# Scan Deployment Notes (FRONT 4) — by Opus 4.8

**Goal:** make the parallel scan the *inference default* in
`reference/moebius_scan_transformer_selective.py`, and settle the
Blelloch-vs-doubling question with a measurement instead of an assertion.

Probe: `src/scan_deployment_probe.py`
Results: `src/scan_deployment_probe_results.json` (canonical = MPS),
plus snapshots `…_mps.json` and `…_cpu.json`.

---

## 1. The measurement (settles Blelloch vs doubling)

All three scans run on the SAME `(a, gamma)`, `B=8 H=4 D=16`, fp32,
median-of-20 wall time (5 warmup, device-synced), torch 2.9.1, Apple MPS.
Correctness gate (max abs err vs sequential `< 1e-5`) **PASSES at every T**
for both parallel scans.

### MPS (the deployment target)

| T    | doubling / seq | blelloch / doubling | doubling beats seq? | blelloch beats doubling? |
|------|----------------|---------------------|---------------------|--------------------------|
| 128  | 2.6–3.2x       | 5.4–13.0x slower    | yes                 | **no**                   |
| 512  | 4.9–6.7x       | 10.2–15.0x slower   | yes                 | **no**                   |
| 1024 | 6.6–12.6x      | 12.6–20.4x slower   | yes                 | **no**                   |
| 2048 | 4.8–7.1x       | 15.9–29.3x slower   | yes                 | **no**                   |
| 4096 | 4.4–7.2x       | 21.5–33.8x slower   | yes                 | **no**                   |

(Ranges span two back-to-back runs; MPS wall time wanders with system load.
The qualitative verdict is identical in both runs.)

### CPU (the fallback target)

| T    | doubling / seq | blelloch / doubling |
|------|----------------|---------------------|
| 128  | 0.81x          | 3.4x slower         |
| 512  | 0.55x          | 2.6x slower         |
| 1024 | 0.33x          | 2.0x slower         |
| 2048 | 0.21x          | 1.4x slower         |
| 4096 | 0.20x          | 1.6x slower         |

### Verdict — straight

- **Blelloch never wins.** Its lower asymptotic *work* (O(T) vs the doubling
  scan's O(T log T)) does **not** translate to wall-time anywhere — not on MPS,
  not on CPU. On MPS it is 5x–34x *slower* than doubling; the gap *grows* with T.
  The cause is exactly what `parallel_scan.py`'s docstring predicted: the
  `A.index_copy(...)` scatter in the up- and down-sweep
  (`parallel_scan.py:170-171, 196-199`) is a serializing, allocation-heavy
  scatter that MPS handles poorly, and it runs 2·log T times. The work-efficient
  variant is correct (`<1e-5`) but **MPS-suboptimal by construction.**
- **Doubling wins on MPS, loses on CPU.** On MPS the pure slice/cat/mul/add
  doubling scan beats the sequential Python loop at every T (~4x–7x in the stable
  band; the T=1024 single-run 12.6x is a slow-seq-sample outlier — treat ~4–7x as
  the number). On CPU there is no parallel hardware to amortize the extra
  O(log T) passes, so the tight sequential loop wins (par is 0.2x–0.8x).

**Ship decision:** doubling (`parallel_linear_scan`) on GPU/MPS; sequential loop
on CPU. Blelloch stays as a verified cross-check, not a deployment path.

---

## 2. The exact wiring change for inference-default

### Where the swap happens

`SelectiveRapiditySqrtScanLayer.forward` (reference file, **lines 141–148**)
calls the **module-global** symbol `sequential_linear_scan` three times:

```python
if self.causal:
    Z = sequential_linear_scan(a, gamma)                 # line 142
else:
    Z_fwd = sequential_linear_scan(a, gamma)             # line 144
    a_rev = torch.flip(a, dims=[1])
    gamma_rev = torch.flip(gamma, dims=[1])
    Z_rev = torch.flip(sequential_linear_scan(a_rev, gamma_rev), dims=[1])  # line 147
    Z = Z_fwd + Z_rev
```

Because the name resolves at *call time* against the module global
(`moebius_scan_transformer_selective.sequential_linear_scan`), this is the single
swap point. The existing `parallel_scan_integration.py::use_parallel_scan` context
manager already exploits exactly this — it sets
`ref.sequential_linear_scan = parallel_linear_scan` and restores on exit. That is a
*measurement* harness, not a *deployment* switch.

### The symbol to introduce

The reference file is frozen (`r--r--r--`), so do **not** edit it. Add a
deployment-default dispatcher *next to* the model — a one-symbol indirection that
picks the right scan from device, with a CPU fallback. Concretely, a small
`scan_runtime.py` (or a few lines at the top of the inference entrypoint):

```python
# scan_runtime.py — deployment default selector
import torch
import moebius_scan_transformer_selective as ref
from parallel_scan import parallel_linear_scan, sequential_linear_scan

def _device_default_scan(a, gamma):
    # a: (B, T, H, D). Pick by where the tensors actually live.
    if a.is_mps or a.is_cuda:          # parallel wins on GPU/MPS (measured)
        return parallel_linear_scan(a, gamma)
    return sequential_linear_scan(a, gamma)   # CPU: tight loop wins (measured)

def enable_parallel_inference():
    """Make the doubling scan the inference default, CPU-safe."""
    ref.sequential_linear_scan = _device_default_scan

def disable_parallel_inference():
    ref.sequential_linear_scan = sequential_linear_scan
```

At inference: `model.eval(); enable_parallel_inference()`. Done — all three call
sites (causal + forward/reverse) now route through the dispatcher, which picks
doubling on MPS/CUDA and the sequential loop on CPU automatically, per the
measured crossover. No edit to the frozen reference; same machine-precision
result (`<1e-5`, gate enforced by `verify_against_sequential`).

### Fallback / safety

- **CPU fallback is automatic** via the `a.is_mps or a.is_cuda` check — on CPU the
  dispatcher returns the sequential loop, which the measurement shows is faster
  there (par is 0.2x–0.8x on CPU).
- **Blelloch is never selected.** It is strictly dominated on both devices.
- **Training already verified.** `parallel_scan_integration.py` confirms forward,
  gradient, and 12-step training identity between the two scans (max |Δloss|
  ~5e-7); the swap is autograd-safe, so the same dispatcher can be left on for
  fine-tuning on GPU if desired. The recommendation here is scoped to inference.
- **Optional constant-γ fast path:** if a deployment freezes `W_gamma` to a
  time-constant forget rate, `constant_gamma_closed_form` (parallel_scan.py:220)
  is the exact O(T²) Toeplitz convolution — not benchmarked here; only relevant
  to the F1 frozen-gate configuration, not the general selective model.

---

## 3. One-line summary

Doubling parallel scan is the inference default on GPU/MPS (~4–7x over the
sequential loop, measured); sequential loop is the CPU fallback (measured to win
there); Blelloch is a verified cross-check that never wins in wall-time because its
`index_copy` scatter dominates its O(T)-work advantage. Wire it via a one-symbol
`ref.sequential_linear_scan = _device_default_scan` dispatcher — no edit to the
frozen reference.
