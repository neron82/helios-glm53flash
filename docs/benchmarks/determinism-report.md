# Helios: Determinismus im Standardpfad

**Ergebnis: Standardpfad für die geprüften Prompts deterministisch; der verlangte Vorher-/Nachher-Nachweis besteht.** Vier unveränderte main-Prozesse divergieren, acht lange und vier kurze Nachher-Prozesse sind jeweils byte-identisch. Alle angeforderten Regressionstests bestehen. Keine Regression >5 %, kein Deployment.

## Auftrag und Aufbau

Basis `4408c57`, isolierter Worktree `helios-determinism`, Branch `determinism`. Referenz-Fix `e5a0c702d687f133b3734dc193ece36c0b05c261` vor dem Port vollständig gelesen. Keine Änderungen im Produktions-Checkout, kein Push/Merge/Deployment.

Build: `PATH=/usr/local/cuda-13.0/bin:$PATH CUDACXX=/usr/local/cuda-13.0/bin/nvcc cmake -B build-determinism -G Ninja`, anschließend `PATH=/usr/local/cuda-13.0/bin:$PATH cmake --build build-determinism -j 4`. CUDA 13.0.88, unveränderte Build-Flags inklusive `--use_fast_math`, arch 86. Eigener Build-Cache; während der Mission durchgehend CUDA 13.0, kein Toolkitwechsel.

`bench/determinism.py` übernimmt die Schutzmaßnahmen aus dem Nacht-Harness: frische Prozesse, sequenzielle Modellläufe, `systemd-run --user --scope -p MemoryHigh=100G -p MemoryMax=110G -p MemorySwapMax=1G`, read-only Root-Mount über bwrap mit Geräten und `/proc`. Default-Census wird gelesen, kann aber nicht geschrieben werden. Die drei effektiven Limits erscheinen in jedem Rohlog. Produktion wird im `finally` über eine unabhängige transiente Unit `helios-prod-restore-det-<epoch>` wiederhergestellt (`Type=oneshot`, `RemainAfterExit=yes`, `KillMode=none`). OmniVoice bleibt aktiv; bge-m3 bleibt unverändert.

Alle Proof-Läufe: Modell A `/home/neron/models/glm53flash_abl`, `gen --cap 262144 --chunk 4096 --tokens 256 --temp 0 --ignore-eos --prefix-snap-mb 0 --prompt-ids …`. Kein Census-/Slot-/Reduktionsschalter. `HELIOS_MTP_TOKENS=1` aktiviert im unveränderten main ausschließlich die ID-Protokollierung; `HELIOS_BENCH_PHASES=1` ist Beobachtung, keine Pfadwahl. Dieselbe Harness-Funktion und dieselben Flags vor/nach Fix. Promptdateien aus der Nachtmission unverändert übernommen, SHA in `provenance.json`.

256 ausgegebene Tokens enthalten das erste Token aus Prefill; 255 Decode-Forwards. Für den Vergleich mit dem vorgegebenen Referenzwert 8,855 tok/s wird dessen Konvention `256000/decode_ms` übernommen. Zusätzlich werden `255000/decode_ms` und `255/(wall_seconds-prefill_ms/1000)` berichtet. Zahlen sind Messwerte; Dezimaldarstellungen im Bericht werden gerundet.

## Fix-Umfang

Port der Reduktionen aus `e5a0c70`, ohne CUDA-Graph-Dateien oder Graph-Schalter. KDA: vier fp32-SUBK-Teilsummen in festen Slots, ein Schreiber addiert in Reihenfolge 0…3. Shared-Add bleibt `add.rn.f32` ohne FTZ, wie Shared-`atomicAdd`; Ausgabe-Casts und Projektionsarithmetik unverändert. MoE: ein Besitzer je gewichteter Contribution, abschließende Summe in ursprünglicher Routing-Rangfolge 0…7. Die inverse Permutation neutralisiert die nichtdeterministische Reihenfolge der ganzzahligen Scatter-Cursor. Der Runner übergibt den Contribution-Puffer immer; der Legacy-Atomic-Modus existiert nur für explizite API-Aufrufer ohne diesen Puffer.

Bestehende Buffer-/Tensor-Dimensionen bleiben unverändert. Der Referenzansatz ergänzt zwei Scratch-Puffer: `[M*8,4096]` fp32 (512 MiB bei M=4096) und `[M*8]` int64 (256 KiB). KDA benötigt zusätzlich SUBK×Head-Dimension Shared-Scratch. Keine Quantisierungs-, Attention-, RoPE-, Sampling-, Toleranz- oder MTP-Änderung. Ein `cuobjdump --dump-ptx` des exakt gemessenen Binarys bestätigt die Add-Semantik: acht explizite `add.rn.f32` ohne FTZ je spezialisiertem KDA-Kernel, `add.rn.ftz.f32` in der geordneten MoE-Summe, entsprechend Shared-/Global-Atomic-Semantik; kompakter Auszug in `ptx-add-semantics.json`. `scatter_add_rows` und `gemm_nt_f16(add)` erhalten einen eindeutigen Besitzer statt eines unnötigen Atomics; separate fp32-Multiply-/Add-Rundung bleibt erhalten.

## Diagnose und Nachweis

Unverändertes main-Binary: `b031d27c630a575d31b8e35eb51565ed1b0967e2c2d311050cea66b747a4e73b`. Vier gültige unabhängige Prozesse, sechs Paarvergleiche: Vorher-Gate **FAIL**. Die Repro ist in diesem Paket nicht ausgeblieben; Erweiterung auf acht Vorher-Läufe ist daher nicht nötig.

| Vorher-Paar | Erster abweichender ID-Index (nullbasiert) |
|---|---:|
| r1 / r2 | 100 |
| r1 / r3 | 69 |
| r1 / r4 | 38 |
| r2 / r3 | 69 |
| r2 / r4 | 38 |
| r3 / r4 | 38 |

Früheste abweichende ausgegebene Entscheidung ist Index 38 (die 39. ID). Da ID 0 aus Prefill stammt, ist dies die Entscheidung nach dem 38. Decode-Forward. Das lokalisiert die sichtbare Tokenabweichung, noch nicht den Beginn der numerischen Variation.
 `HELIOS_DET_PLANES=1` beobachtet optional Gerätebytes als FNV-1a-Zeilenfingerprints an Layer-Grenzen und vor/nach Rekurrenz. Das ist ein Lokalisierungswerkzeug, kein Ersatz für die exakte Token-Byteprüfung. Ohne diese Variable entstehen keine Host-Kopien oder zusätzlichen Synchronisationen.

### Erste numerische Abweichung

Zwei unabhängige Originalnumerik-Prozesse mit Beobachtung: Binary `0422e62404f6c9a083018934b6833aaa8fb494f2e5c7bea26a396da83737b6fc`. Je **678** benannte Beobachtungen aus dem vollständigen 4096er Prefill und dem ersten Decode-Forward. Die erste beobachtete Abweichung ist **Prefill, Layer 0, `rec_output`, Zeile 3**, 1213/4096 Ausgaberows unterschiedlich. Alle Zeilenfingerprints der fünf davor beobachteten Planes stimmen überein: `layer_input`, `conv_out`, `gate`, `beta`, `rec_before`. Der abschließende fp32-Rekurrenzzustand variiert ebenfalls. Im ersten Decode-Forward (Position 4096) ist der früheste beobachtete Unterschied bereits der gespeicherte **L0-`rec_before`-Zustand, Zeile 0**, 7253/8192 Zeilen unterschiedlich. Die Prefill-Variation wird also in Decode übernommen.

Zuständiger Kernel: `cuda_recurrent_gated_delta_rule_kernel_128<false,4,true>` (Head-Dimension 128, channelwise=true). Unverändertes main: Shared-`atomicAdd` in `src/cuda/aux/gdn.cu:682` und `:711`; generic Pendants `:479` und `:515`. Die erste gemessene Abweichung liegt innerhalb dieser Rekurrenz, nicht vor ihr. Die vier SUBK-Threadgruppen addieren ihre fp32-Teilsummen in einer vom Warp-Fortschritt abhängigen Reihenfolge. Die neue feste Summe steht im portierten Kernel bei `gdn.cu:705` und `:740`.

Dies lokalisiert die erste **beobachtete** Variation; der genaue erste abweichende fp32-Add innerhalb des 4096-Schritte-Kernels wurde nicht einzeln instrumentiert. BF16-Ausgaben können frühere fp32-State-Unterschiede verdecken. Die Zeilenfingerprints sind kein mathematischer Ausschluss von Hash-Kollisionen und kein 139-Plane-Bytebeweis. Das graphabhängige 139-Plane-Testgerüst wurde gelesen, aber nicht samt Graph-Maschinerie übernommen; die minimale benannte Plane-/Zeilendiagnose vermeidet diese Abhängigkeit. Die Original-Logs und `localization.json` / `localization-first-layer.json` dokumentieren den Befund.

Die weitere ungeordnete MoE-Ausgangssumme (`main` `hadamard_inner.cuh:469–472`) wird im gleichen minimalen Referenzsatz beseitigt. Eine separate Vollmodell-Abtragung des MoE-Anteils wurde nicht gemessen; der empirisch früheste Befund ist KDA. Die MoE-Summe wird strukturell durch disjunkte Beiträge und einen Besitzer je Ausgabe ersetzt (`glue2.cu:297`).

### Langer Vorher-/Nachher-Nachweis

Fix-Commit `c2a9a07`; eingefrorenes Nachher-Binary **`31bf6d086dd8a292cfbed91d71307353a0a46d82ac3692bb983425d442889dfb`**. Acht unabhängige Standardprozesse: **28/28 Paarvergleiche ohne abweichenden Index**, alle gespeicherten ID-Dateien byte-identisch, je 256 IDs, Exit 0, keine Degraded-/CUDA-/pageable-Warnung. Dieselbe Capture-Funktion wie vor dem Fix; das gemeinsame Vergleichswerkzeug akzeptiert ausschließlich exakte Dateibytes und ID-Gleichheit. Vorher FAIL, nachher PASS.

| Lauf | SHA-256 der ID-Datei | tok/s (`256000/decode_ms`, gerundet) |
|---|---|---:|
| before-A-4096-r1 | `1a34db2f989eb79ea3874fef54810001960a4a156773ef95f6b731c8bb389a70` | 8.728205 |
| before-A-4096-r2 | `9fd03bde6a9f36fdabc338130f78d66b6bf3aba0170c93e89ad904491c898db7` | 8.867162 |
| before-A-4096-r3 | `57d8bd87ef80a5dda8e9cce50f39c120ac8938ca26629facbc6b5092ac3b6497` | 8.850991 |
| before-A-4096-r4 | `d9a342e6ddadf9b4e085ee32108ebc6041c3dcc39d0ab5b0c33fc9f0f7f5262f` | 8.885179 |
| after-A-4096-r1 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.890912 |
| after-A-4096-r2 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.909565 |
| after-A-4096-r3 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.809888 |
| after-A-4096-r4 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.901358 |
| after-A-4096-r5 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.874194 |
| after-A-4096-r6 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.904001 |
| after-A-4096-r7 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.905381 |
| after-A-4096-r8 | `cfefe2d69a6fff2333babd6d56a772557a41e9bddcc691f33c1b817cc639793d` | 8.908470 |

Nachher-Median **8.902680 tok/s** (acht Läufe), min/max 8.809888/8.909565. Gegen die vorgegebene Referenz 8,855 tok/s: **+0.5384%**. Keine Regression >5 %. Sekundär: Median 8.867903 Decode-Forwards/s bzw. 8.844939 ausgegebene Decode-Tokens/s einschließlich Sampling/Ausgabe. Vergleich zum Median der vier neuen Vorher-Läufe (8.859076): +0.4922 %. Der erste Vorher-Request überlappte mit CPU/CUDA-Kompilation des Folge-Builds; die vorgegebene unabhängige Referenz bleibt der Hauptvergleich. Alle Nachher-Requests liefen ohne parallelen Build.

Routen/Residency können mit der neuen kanonischen Reihenfolge variieren; Durchsatz ist eine Vollmodellmessung und keine isolierte Kernel-Zeit. Kein aufgelöster Performancegewinn wird aus dem kleinen Unterschied behauptet. Maximal gemessener Nachher-Peak-RSS: 100.973 GB (dezimal). VRAM-Minima sind 1-s-Samples, keine Garantie des exakten kurzfristigen Peaks; alle Samples liegen lokal in den benannten Rohdateien.

### Kurzer Zähl-Prompt und Gesamtkriterium

`prompt-identity.json`, 51 Prompttokens, unveränderte Generierungsflags. Vier frische Nachher-Prozesse mit dem gleichen eingefrorenen Binary:

| Lauf | SHA-256 der 256-ID-Datei | Exit |
|---|---|---:|
| identity-A-r1 | `204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86` | 0 |
| identity-A-r2 | `204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86` | 0 |
| identity-A-r3 | `204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86` | 0 |
| identity-A-r4 | `204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86` | 0 |

Alle sechs Paare byte-identisch. **Die neue kanonische Liste bleibt exakt die alte Referenz `204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86`; kein abweichender Index.** Die alte Liste ist unverändert in `identity-reference-ids.json` archiviert.

`python3 bench/summarize_determinism.py`, Exit **0**: `summary.proven=true`, `identical_flags=true`, `census_unchanged=true`. Das gemeinsame Gate akzeptiert nur: Vorher FAIL bei mindestens vier Prozessen; Nachher PASS mit mindestens acht langen plus vier kurzen gültigen Prozessen; ein einziges Nachher-Binary; identische lange Flags; jedes effektive Scope-Limit korrekt; unveränderte Census-Hashes. Kein Toleranzvergleich ersetzt die ID-Dateigleichheit. Hashes sind SHA-256 der konkreten JSON-Dateibytes (Python-JSON mit Leerzeichen und abschließendem Newline).

## Regression und Produktion

CPU-Tests vor/nach Fix: die fünf angeforderten CTests bestehen; Tokenizer A/B je 28/28. GPU-Tests bestehen im letzten Messpaket. `dsa_topk_order` existiert in main nicht und wird nicht erwartet. Der optionale lange Modell-B-Proof wurde wegen zusätzlicher Produktionsausfallzeit nicht gefahren; nur die angeforderten Tokenizer-Regressionen für B werden beansprucht.

### Testausgabe

Alle Befehle und Log-Hashes: `bench/results/determinism/test-results.json`. CTests und Tokenizer wurden auch vor dem Fix gefahren. Die drei GPU-Tests laufen sequenziell nach dem letzten Modellprozess, während Produktion gestoppt ist, und ihre Exitcodes werden im Harness jeweils auf 0 geprüft.

```text
Internal ctest changing into directory: /home/neron/projects/new_engine/helios-determinism/build-determinism
Test project /home/neron/projects/new_engine/helios-determinism/build-determinism
    Start 1: expert_layout
1/5 Test #1: expert_layout ....................   Passed    0.01 sec
    Start 2: prefix_policy
2/5 Test #2: prefix_policy ....................   Passed    0.00 sec
    Start 3: utf8
3/5 Test #3: utf8 .............................   Passed    0.00 sec
    Start 4: chat_protocol
4/5 Test #4: chat_protocol ....................   Passed    0.00 sec
    Start 5: vision_support
5/5 Test #5: vision_support ...................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   0.02 sec
```

Tokenizer, beide Checkpoints, jeweils vor/nach:

```text
TOKENIZER: 28/28 pass (ALL PASS)
```

`build-determinism/attn/test_attn_parity`, Exit 0, unveränderte Paritätsgrenzen (letzte Zeilen):

```text
mla_sparse_decode relerr 2.80e-04 PASS
mla_dense_prefill relerr 1.56e-04 PASS
mla_sparse_decode K=1 vs K=2: relerr 2.00e-04, 0/98304 beyond 2e-2 PASS
indexer_score mma vs scalar: 0/28800 differ (max|d|=3.05e-05), mask mismatches 0, -inf 16596/16596 PASS
ATTN PARITY: ALL PASS
```

`build-determinism/aux/test_aux`, Exit 0:

```text
RMS norm max error: 0.000244141 (half precision expected <0.01)
PASS: RMS norm
SILU mul max error: 0.00195312 (half precision expected <0.01)
PASS: SILU mul
PASS: DSA topk
All aux parity tests passed.
```

`build-determinism/helios_deterministic_reductions_test`, Exit 0: alle **12** generischen/spezialisierten Rekurrenz-Konfigurationen (64/128, scalar/channelwise, history/direct, 1/7 Schritte) bestehen jeweils **16** restaurierte Wiederholungen mit exakten BF16-Ausgabe-/FP32-State-Bytes. Dazu acht inverse MoE-Permutationen mit FP32-Auslöschung sowie Scatter-/GEMM-Besitzprüfung. Letzte Zeilen:

```text
PASS recurrence dim=128 channelwise=1 history=0 steps=7 repeats=16 zero differing bytes
PASS recurrence dim=128 channelwise=1 history=1 steps=1 repeats=16 zero differing bytes
PASS recurrence dim=128 channelwise=1 history=1 steps=7 repeats=16 zero differing bytes
PASS MoE fixed routing order through 8 permutations, fp32 cancellation result=2
PASS unique scatter and tiled GEMM additions
```

### Artefakte und Wiederholung

Gemeinsames Capture-Harness: `bench/determinism.py`; exakter Vergleich/Throughput: `bench/summarize_determinism.py`; minimaler Originalnumerik-Diagnose-Build: `bench/prepare_diagnostic.py` (auf sauberem main **nach** Sicherung des unveränderten Binarys ausführen); Vergleich: `bench/compare_det_planes.py`. Die Diagnose fügt nur opt-in Beobachtung hinzu. Produktions-Endprüfung: `bench/verify_determinism_production.py` (read-only).

Messbefehle auf den bereits gesicherten Builds:

```bash
python3 bench/determinism.py before
python3 bench/determinism.py diagnostic
python3 bench/determinism.py after --start 1 --count 4
python3 bench/determinism.py after --start 5 --count 4
python3 bench/determinism.py identity --count 4
python3 bench/summarize_determinism.py
python3 bench/verify_determinism_production.py
```

Diese Harness-Befehle verwalten die autorisierten Stop-/Restore-Fenster selbst. Kein zweiter Modelllauf parallel. Die Diagnose nutzt Originalnumerik-Binary und zusätzlich `HELIOS_DET_PLANES=1`; sie zählt nicht zum Nachher-Standardproof.

Committe Rohlogs aller 4+8+4 Generierungsprozesse, alle ID-Listen, die beiden Diagnose-ID-Listen, Test-/Build-/Orchestrierungslogs, Scope-Zeilen, Prompt-/Census-/Binary-Hashes und strukturierte Vergleichsergebnisse. Große Diagnose-Logs `logs/diagnostic-before-r1.log` und `logs/diagnostic-before-r2.log` (je rund 37 MiB), sowie sämtliche `logs/*-vram.jsonl` und der vollständige PTX-Dump `ptx-final.txt` bleiben lokal und **nicht committed**; die vollständigen Namen, Größen und SHA-256 stehen in `uncommitted-raw-files.json`. Die Diagnosezusammenfassungen sind committed. Binaries und Build-Produkte bleiben ebenfalls uncommitted in `build-determinism/`: `helios-before`, `helios-before-diagnostic`, `helios`; deren SHA-256 stehen in den Binary-Manifesten. Der read-only Audit nach dem Ergebniscommit liegt lokal unter `final-audit-after-commit.json`.

Beweisgrenze: zwei konkrete Prompts, Modell A, obige Flags und Hardware/Toolkit. Kein neuer 139-Plane-Bytegleichheitsanspruch, kein langer Modell-B-Proof, keine CUDA-Graph- oder Live-Vision-Abnahme. Diese optionalen Erweiterungen sind für den verlangten 8+4 Token-Nachweis nicht beansprucht. Das Produktions-Binary bleibt unverändert.

### Produktions-Ausfallfenster und Endzustand

Alle Zeiten am **2026-10-07**, Europe/Berlin (**UTC+02:00**). Konservativ erfasst vom Start des Stop-Aufrufs bis zur bestätigten Gesundheit nach dem sofort eingeleiteten Restore. Sekundenauflösung; keine hochauflösende Beobachtung des exakten Listener-Übergangs. Zwischen den Paketen war Produktion wieder gesund.

| Paket | Stop begonnen | Restore begonnen | Gesundheit bestätigt | Erfasstes Fenster |
|---|---|---|---|---:|
| before | 07:35:31 | 07:44:46 | 07:46:18 | 10:47 min |
| diagnostic | 07:47:01 | 07:53:30 | 07:55:00 | 7:59 min |
| after 1–4 | 07:55:21 | 08:04:31 | 08:06:01 | 10:40 min |
| after 5–8 | 08:06:09 | 08:15:18 | 08:16:49 | 10:40 min |
| identity | 08:17:44 | 08:25:57 | 08:27:26 | 9:42 min |

Summe der konservativ erfassten Fenster: **49:48 min**. Rohzeitstempel in `outages.json`; vollständige Restore-Antworten/Original-Hashes in den Orchestrierungslogs. Die Restore-Unit wurde nach jedem Paket ohne dazwischenliegende Arbeit gestartet; die übrige Ausfallzeit entsteht durch unabhängige Modellstarts, Generierung und Produktionsstart.

Endprüfung **2026-10-07T08:28:08.713685+02:00**, Exit **0**:

- Produktion PID **713384**, eigener Cgroup `helios-prod-restore-det-1791354357148088482.service`, außerhalb des Worker-/Benchmark-Scopes.
- Live-Binary SHA-256 **`1a9cf7dc0557207a717746d796d224a5a94d4e99bf837c95f839061752c7a813`**, exakt der vorgegebene Originalhash.
- `/health`: `{"status":"ok"}`; `/v1/models`: **glm-5.3-flash-exl3**, context_length **262144**.
- Restore-Unit: **Type=oneshot**, **RemainAfterExit=yes**, **KillMode=none**, **MemoryMax=infinity**, aktiv.
- **omnivoice.service active**, Health healthy, ready=true, model_loaded=true.
- **bge-m3 PID 26170** unverändert, Listener **7777** gehört diesem PID, Health `{"status":"ok"}`.

Keine Launcher-/Systemd-Konfiguration geändert, kein Produktions-Checkout bearbeitet, kein Push/Merge/Deployment. Ein weiterer read-only Audit wird nach dem Ergebniscommit als letzte Prüfung ausgeführt und lokal unter `final-audit-after-commit.json` erhalten. Offen bleibt nur der optionale lange Modell-B-Proof; alle verpflichtenden Punkte sind abgeschlossen.
