# CODE-MAP

Zwei parallele Pipelines:
- `analyze_residuals/` + `analyze_n_ens/`: Fenster via **M / years** (kein --tau), `--time_mode {linear,log}`, `--save`.
- Root `ssmlearn_*` / `harmonic_*` / `SSM/`: Fenster via M/years bzw. tau, `--out`/`--save-prefix`.

**Signale:** Residuum r = log(p/PLB), PLB = QuantReg q=0.01 auf log p ~ log t. Exponent n(t) = zentrierte log-log-Steigung, Ensemble w=90..180 d.
**Defaults überall:** M=35, years=3.77 (→ τ=40 d, W=1360 d), start_idx=1164, ziel.csv.
⚠ = hartkodierter Output-Pfad (Überschreib-Gefahr).

Konfig-Abhängigkeit der SSM-Zahlen: d=4/poly=1 (linear clock) → Master 3.86 a, Sub 2.04 a.
d=3/poly=2 (13_cli, beide Clocks) → linear Master 4.00 a; log Master λ=1.983. Exponent d=3 → linear 3.44 a, log λ=1.24.
13_2_cli-Defaults (dim9/dim5) liefern ANDERE Master (7.9 a / λ=1.14) — für λ≈2-Aussagen 13_cli mit ssm_dim=3 verwenden.

---

## Haupt-Pipeline: Skript → Bild

Kommandos aus dem jeweiligen Skript-Ordner, `MPLBACKEND=Agg`, Ziel via `--save`/`--out`.

### `btc_scale_inv_PLB.py` (in `/home/hz/Data/`)
Scale-Invarianz-Test, 10%-Quantil-Boden (hartkodiert Z.17).

![scale invariance](figs/fig_scaleinv.png)

### `surrogates/` M-Scan (Artefakt `m_scan.png`)
M-Optimierung: W=1376 d fix, τ=W/(M−1); Optimal-Band M∈[32,60], M=35 = kleinstes M im Band.

![m scan](figs/fig_m_scan.png)

### `analyze_residuals/14_cli_phase3d.py`
3D-Phasenraum des Residuums. `--time_mode linear --no-show --save <ziel>`

![residual phase space linear](figs/fig_res_phase3d.png)

`--time_mode log`:

![residual phase space log](figs/fig_res_phase3d_log.png)

### `analyze_n_ens/14_cli_phase3d.py`
3D-Phasenraum des Exponenten. `--time_mode linear --windows 180 --no-show --save <ziel>` (ssm_dim=3)

![exponent phase space linear](figs/fig_exp_linear.png)

`--time_mode log --ssm_dim 3 --windows 180`:

![exponent phase space log](figs/fig_exp_log.png)

### `ssmlearn_manifold.py`
2D-SSM als 3D-Fläche. `--M 35 --years 3.77 --ssm_dim 2 --poly_degree 3 --no-show --out <ziel>`

![ssm manifold](figs/fig_res_ssm_manifold_tau40.png)

### `ssmlearn_res.py`
SSMLearnPy-Diagnostik, Eigenwerte im CLI-Output. `--M 35 --years 3.77 --ssm_dim 4 --poly_degrees 1,2,3 --no-show --out <ziel>`

![ssm d4 diagnostics](figs/fig_ssm_final_tau40.png)

### `harmonic_test_phase.py`
Phase-Locking Residuum. `--ssm_dim 4 --poly_degree 2 --loc --no-show --out <ziel>` — Verdikt-Zeile aus suptitle entfernen (Kopie, nicht Original)

![phase locking residual](figs/fig_phase_res.png)

### `harmonic_test_n_phase.py`
Phase-Locking Exponent. `--M 35 --years 3.77 --ssm_dim 4 --poly_degree 1 --loc --no-show --out <ziel>` — Verdikt + windows-Aufzählung strippen

![phase locking exponent](figs/fig_phase_exp.png)

### `analyze_residuals/13_cli_lambda2_visualize.py`
Linear- vs. log-Clock-SSM, Residuum (ssm_dim=3, poly=2 → λ=1.983≈2). `--no-show --save <ziel>`

![ssm res log](figs/fig_ssm_res_log.png)

### `analyze_n_ens/13_cli_lambda2_visualize.py`
Dito Exponent (λ≈1.24). `--ssm_dim 3 --no-show --save <ziel>`

![ssm exp log](figs/fig_ssm_exp_log.png)

### `SSM/res/surrogate_ssm.py`
IAAFT-Surrogate-Test (p_rank=0.02, 2.9σ). ⚠ Output fix `surrogate_ssm_fig1/2.png`

![surrogate test](figs/fig_surrogate.png)

### `attractor_scan_recurrence.py`
RQA-τ-Scan (%DET≈0.98 ∀τ). ⚠ Output fix `recurrence_scan.png`

![recurrence scan](figs/fig_recurrence.png)

### `attractor_compare_raw_vs_ssm-linear.py`
2×2-Vergleich der Embeddings: M=6 (Perrenod-artig) vs. M=35+PCA, roh vs. +PCA — linear-time sampled, Halving-Cycle-Farben. ⚠ Output fix `compare_raw_vs_ssm-linear.png`

![compare raw vs ssm linear](figs/compare_raw_vs_ssm-linear.png)

### `attractor_compare_raw_vs_ssm-logtime.py`
Dito, log10-time sampled (kontinuierliche log10(day)-Colorbar). ⚠ Output fix `compare_raw_vs_ssm-logtime.png`

![compare raw vs ssm logtime](figs/compare_raw_vs_ssm-logtime.png)

---

## analyze_residuals/ (Residuum)
| Skript | Zweck | Clock | Output |
|---|---|---|---|
| 02_cli_harmonics.py | PSD je PC | --time_mode | nur show |
| 03_cli_phase.py | Phasen-Kopplung 2φ1−φ2 | --time_mode | nur show |
| 03b_cli_phase_all.py | Phase-all, log fix | log | --save (auto) |
| 04_cli_scaling.py | Amplituden-Skalierung | --time_mode | nur show |
| 05_cli_scan.py | SSM-Sweep dim×poly | --time_mode | nur show |
| 10_cli_ssm_learn.py | SSMLearn Slave-Test | --time_mode | nur show |
| 13_cli_lambda2_visualize.py | linear-vs-log SSM (λ2) | beide | --save |
| 13_2_cli_lambda2_visualize.py | dito, 2. Layout (dim9) | beide | --save |
| 14_cli_phase3d.py | 3D-Phasenraum | --time_mode | --save |
| 15_cli_resample_control.py | Prewhiten-Kontrolle | beide | savefig auto |

## analyze_n_ens/ (Exponent)
13_cli / 13_2_cli / 14_cli / 03b analog; zusätzlich `--windows` (default [180]), `--log_uniform_mean/_both`.

## Root (Auswahl mit Output-Flag)
| Skript | Zweck | Output-Flag |
|---|---|---|
| ssmlearn_res.py | SSMLearnPy Residuen (Eigenwerte) | --out |
| ssmlearn_manifold.py | 2D-SSM als 3D-Fläche | --out |
| harmonic_test_phase.py / _n_phase.py | Phase-Locking res/exp | --out |
| harmonic_test_backbone/_bicoherence/_slave.py | Backbone/Bikohärenz/Slave | --out |
| attractor_verify.py / _surrogates.py | kNN-Kreuzcheck, Kaplan-Glass | --save |
| lpplattr02_poincare_native/_strobe.py | LPPL-Poincaré | --save-prefix |
| attractor_scan_recurrence.py | RQA-τ-Scan | ⚠ fix |
| attractor_compare_raw_vs_ssm-{linear,logtime}.py | 2×2 Vergleich M=6 vs M=35 | ⚠ fix |
| attractor_scan_m.py / _mutual.py / attractor*.py (alt) | Scans/ältere Attraktoren | nur show |

## SSM/res|n|raw (zweite Pipeline, tau-basiert)
| Skript | Zweck | Output |
|---|---|---|
| SSM_res.py | volle Residuen-SSM (4 Figs) | --save-prefix |
| SSM_res_free.py | pure TDE/intrinsic dim | --save-prefix |
| 13_cli_manifold.py (res+n) | SSM-Manifold 3D, --time_mode | nur show |
| 11/12_cli (res) | Embedding-Moden / intrinsic dim | nur show |
| surrogate_ssm.py | IAAFT-Surrogate | ⚠ fig1/fig2 fix |
| SSM/raw/SSM_raw.py | Roh-Preis-SSM, --log/--lin | --save-prefix |
| 02–10_cli (SSM/res) | synthetisches LPPL-System | nur show |

## surrogates/
| Skript | Zweck | Output |
|---|---|---|
| surrogates_theiler_test.py / _n.py | Theiler-Tests res/exp (n_surr=999) | ⚠ fix |
| surrogates_n_IAAFT.py (+_compute/_plot) | IAAFT Exponent, --loc/--norm/--cat | show/JSON |
| recurrence_tau_scan.py | RQA-τ-Auswahl | nur show |
| m_scan.png | M-Optimierung (Artefakt) | — |
