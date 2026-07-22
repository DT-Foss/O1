# MISSION: POS — Plasticity-on-Surprise als Betriebsmodus, gemessen in 48h

Du arbeitest im Repo DT-Foss/O1 (lies zuerst: README.md, src/streaming_train.py,
src/closed_loop.py, src/verify_billion.py, reference/moebius_scan_transformer_selective.py).
Alles Nötige existiert bereits im Repo; deine Aufgabe ist Integration + Messung,
keine Neuarchitektur. Konventionen des Repos übernehmen: src/ = Code,
results/ = JSON-Evidenz, plots/ = PNG, analysis/ = These. Code/Doku auf Englisch.

## DIE THESE (was du beweist)
Gradient-on-Surprise: Forward läuft immer (no_grad, Z-Carry), Backward nur auf
Spannen, deren Surprise über der Schwelle liegt. Erwartung: der gated Arm holt
den Großteil des Lernens des Vollgradient-Arms bei einem Bruchteil der
Gradient-Tokens — bei flacher RSS, auf einem CPU-Prozess, der nie neu startet.

## AUFBAU (ein Prozess, drei Arme, identischer Datenstrom)
Ein Runner `src/pos_run.py`, C4 via HF streaming (wie scale_to_a_billion.py),
jeder Chunk wird derselben Reihenfolge nach an alle drei Arme gefüttert
(Tensor klonen — exakte Datenfairness):
  A1 forward-only   (Kontrolle: kein Lernen)
  A2 full-gradient  (Kontrolle: Standardrezept, jeder Chunk backward)
  A3 surprise-gated (backward nur wenn Chunk-Mean-Surprise > q80 eines
                     Rolling-Fensters; Ziel ~20% Gradient-Tokens; q ist Parameter)
Alle Arme: StreamingNoPELM, gleiche Init (Seed 42), detach-Carry an jeder
Chunk-Grenze (das ist der exakte Pfad aus streaming_train.py, grad-cos 1.0),
CPU, Threads wie im Repo gecappt, psutil-RSS-Logging.

Dazu der INDEX-LOOP auf A3 (Mechanik aus closed_loop.py wiederverwenden):
Surprise-Spike ⇒ Span (±32 Tokens) unter n-Gramm-Schlüssel in results/pos_index.jsonl.
Bei Wiederauftreten eines Schlüssels ≥2h später: gepaarter Test vom selben
Pre-Gap-State — Injection des gespeicherten Spans vs. Random-Span, Δ-Surprise
loggen. Das ist der committete Falsifier, nur über Wallclock gestreckt.

RESET-ZWILLING: bei T+24h forkt der Runner A3 → A3R, nullt Z + Optimizer-State,
läuft parallel weiter. Misst, was der Neustart an Warmup neu bezahlt.

## ARBEITSPAKETE (Task-Tool, Subagents)
WP1 (zuerst, solo): Harness — pos_run.py mit drei Armen, Gating, Checkpoints
    (alle 10 Min, tmpfile + os.replace, resume-fähig), status.json-Heartbeat.
WP2 (parallel nach WP1): Index-Loop + Injection-Test + Zwillings-Fork.
WP3 (parallel nach WP1): pos_analyze.py (Plots: Loss vs. Streamed-Tokens,
    Loss vs. GRADIENT-Tokens [der Kernplot], RSS-Timeline, Surprise-Timeline,
    Injection-Paired-Bars, Zwillings-Recovery) + verify_pos.py nach dem Muster
    von verify_billion.py: prüft jede Claim-Zahl gegen die JSONs, exit 0.
Du integrierst, führst Reviews der Subagent-Ergebnisse selbst durch.

## BUILD-GATES (Qualität, vor dem Langlauf — dann losfahren)
G1 Parity: A2-Pfad reproduziert streaming_train.py-Verhalten (Loss fällt, RSS flach).
G2 Determinismus: 2×3-Min-Smoke mit Seed 42 ⇒ bitgleiche status.json-Metriken.
G3 Gating live: Smoke zeigt A3-Gradient-Token-Anteil im Band 15–30%, sonst q justieren.
Nach G1–G3: Langlauf detached starten —
  nohup python -u src/pos_run.py --hours 40 > results/pos_run.log 2>&1 &
Maschinenschutz im Runner: pausieren bei RSS-Gesamt > 12 GB oder Disk < 5 GB.

## ZIELBILD (Orientierung, kein Stop-Kriterium — du denkst selbst)
A3 erreicht ≥75% der Loss-Verbesserung von A2 bei ≤25% der Gradient-Tokens;
Injection senkt Follow-on-Surprise (Mehrheit der gepaarten Probes), Random nicht;
RSS aller Arme in einem ±0.1-GB-Band; A3R zahlt sichtbaren Warmup, A3 nicht.
Weicht eine Zahl ab: charakterisiere die Kurve, justiere q, lass weiterlaufen —
Abweichungen sind Messpunkte, keine Fehlschläge. Frozen-Eval: fixe 200k-Token-
WT-2-Val-Scheibe, alle 15 Min, no_grad.

## ABSCHLUSS DEINER SESSION (Phase A)
1. Alles committen (lokal, kein push), aussagekräftige Messages.
2. NEXT.md schreiben: der eine Befehl für T+48h:
   claude --model claude-opus-4-8 --dangerously-skip-permissions \
     -p "Führe Phase B aus NEXT.md aus: pos_analyze.py laufen lassen, Plots
     erzeugen, verify_pos.py grün machen, analysis/POS_THESIS.md (≤1 Seite,
     Zahlen aus results/) schreiben, committen."
3. Kurzreport auf stdout: was läuft, PID, wo status.json liegt.
