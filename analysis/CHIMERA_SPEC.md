# CHIMERA — the complete organism (MS6 spec, v0)

*Written day 4 ~07:30 by the lead. Every design decision below is the measured
answer to a scored prediction — no organ ships on taste. This spec is the F1
locking experiment: one process in which the surprise calculus makes every
plasticity and memory decision.*

## The organism (one process, one stream)

```
wake:    stream C4 chunks  ->  no-grad forward (Z-carry)
gate:    backward only if chunk-NLL > rolling q75 of own history   [P1/P22: ratio ~1 at ~25% grad tokens; P28: FIXED quantile, not rate-homeostat]
store:   surprise spikes -> span store (recurrence-keyed)          [P19: dreams lose to stored spans — storage STAYS]
remind:  recurring stored keys -> in-sequence injection            [P18/MS1: reliable-index statistics; arbitration is a LATER moonshot]
sleep:   replay stored spans, freshness-weighted, ONLY while the
         measured dividend > 0 (held-out delta per sleep block)    [M4 life curve; P20: recovery-phase overdose — the monitor gates EVERY phase]
operate: weight updates / widening / moment migration allowed LIVE [P23/P24/P26/P27: fast path heals in 256 tok; content survives at 1.000]
```

## Organ dimensioning (all measured)

| organ | setting | evidence |
|---|---|---|
| gate | q=0.75 fixed, window 500, min 100 | P1 interim 1.002 @410M; P28 killed auto-q |
| span store | keyed spans, freshness-weighted sampling | MS2 kill + M4 life curve |
| sleep | volume-coupled budget, dividend monitor (stop at ≤0), applies in recovery too | P15/P20 |
| reminder | in-sequence injection on recurrence, reliable statistics | P18 (0.99–1.00 read when trained-with) |
| holo phase bank | scarce corner only: P_max ≤ 16, d_head-bank ≤ 64 | P32 rent map (valley elsewhere) |
| persistence prosthesis | eval-time clamp+refresh at deploy | P29/P30 (interaction, knee 2176+) |
| architecture | GSSM-Selective; family-generic clause | P22 (S6 transfer 0.98; GSSM leads 0.156) |

## The measurement (P33)

Benchmark = the MS3 shock protocol (C4→code→C4), headline metric CONTINUAL:
forgetting / plasticity / recovery, plus held-out trajectory. Arms at matched
total gradient tokens:
- (i) CHIMERA (all organs, as above);
- (ii) best single-organ regime measured so far (R3: gate+sleep — forgetting
  +0.246, plasticity +0.366, recovery +0.234 at full scale);
- (iii) full-gradient baseline (R1: forgetting +0.667);
- (iv) ablation: CHIMERA minus reminder; (v) CHIMERA minus dividend monitor
  (fixed-cadence sleep) — the two organs whose marginal value is untested in
  composition.

P33 (registered with this spec): (a) CHIMERA forgetting ≤ R3's at ≥ R3's
plasticity; (b) CHIMERA recovery beats R3's +0.234 (the dividend monitor must
fix the recovery-overdose P20 found); (c) each ablation is worse on at least
one axis than full CHIMERA (no dead organ rides along); (d) the whole beats
the sum: no single-organ arm dominates CHIMERA on all three axes.

## Open questions (not in v0)

Arbitration under unreliable indices (MS1 conflict finding); auto-curiosity
(P28's surprise-LEVEL target variant); cross-organism spans in-flight (P31
measured the offline case); growth mid-run (licensed by P24/P27, not exercised
in v0).
