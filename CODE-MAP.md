# CODE-MAP

Two parallel pipelines:
- `analyze_residuals/` + `analyze_n_ens/`: window via **M / years** (no --tau), `--time_mode {linear,log}`, `--save`.
- Root `ssmlearn_*` / `harmonic_*` / `SSM/`: window via M/years or tau, `--out`/`--save-prefix`.

**Signals:** residual r = log(p/PLB), PLB = quantile regression q=0.01 on log p ~ log t. Exponent n(t) = centred log-log slope, ensemble w=90..180 d.
**Defaults everywhere:** M=35, years=3.77 (→ τ=40 d, W=1360 d), start_idx=1164, ziel.csv.
⚠ = hard-coded output path (overwrite risk).

Config dependence of the SSM numbers: d=4/poly=1 (linear clock) → master 3.86 y, sub 2.04 y.
d=3/poly=2 (13_cli, both clocks) → linear master 4.00 y; log master λ=1.983. Exponent d=3 → linear 3.44 y, log λ=1.24.
The 13_2_cli defaults (dim9/dim5) yield DIFFERENT masters (7.9 y / λ=1.14) — for λ≈2 statements use 13_cli with ssm_dim=3.

---

## Main pipeline: script → figure

Commands are run from the script's folder, `MPLBACKEND=Agg`, target via `--save`/`--out`.

### `btc_scale_inv_PLB.py` (in `/home/hz/Data/`)
Scale-invariance test, 10% quantile floor (hard-coded, line 17).

![scale invariance](figs/fig_scaleinv.png)

### `surrogates/` M scan (artifact `m_scan.png`)
M optimisation: W=1376 d fixed, τ=W/(M−1); optimal band M∈[32,60], M=35 = smallest M inside the band.

![m scan](figs/fig_m_scan.png)

### `analyze_residuals/14_cli_phase3d.py`
3D phase space of the residual. `--time_mode linear --no-show --save <target>`

![residual phase space linear](figs/fig_res_phase3d.png)

`--time_mode log`:

![residual phase space log](figs/fig_res_phase3d_log.png)

### `analyze_n_ens/14_cli_phase3d.py`
3D phase space of the exponent. `--time_mode linear --windows 180 --no-show --save <target>` (ssm_dim=3)

![exponent phase space linear](figs/fig_exp_linear.png)

`--time_mode log --ssm_dim 3 --windows 180`:

![exponent phase space log](figs/fig_exp_log.png)

### `ssmlearn_manifold.py`
2D SSM as a 3D surface. `--M 35 --years 3.77 --ssm_dim 2 --poly_degree 3 --no-show --out <target>`

![ssm manifold](figs/fig_res_ssm_manifold_tau40.png)

### `ssmlearn_res.py`
SSMLearnPy diagnostics; eigenvalues in the CLI output. `--M 35 --years 3.77 --ssm_dim 4 --poly_degrees 1,2,3 --no-show --out <target>`

![ssm d4 diagnostics](figs/fig_ssm_final_tau40.png)

### `harmonic_test_phase.py`
Phase locking, residual. `--ssm_dim 4 --poly_degree 2 --loc --no-show --out <target>` — strip the verdict line from the suptitle (work on a copy, not the original)

![phase locking residual](figs/fig_phase_res.png)

### `harmonic_test_n_phase.py`
Phase locking, exponent. `--M 35 --years 3.77 --ssm_dim 4 --poly_degree 1 --loc --no-show --out <target>` — strip verdict + windows enumeration

![phase locking exponent](figs/fig_phase_exp.png)

### `analyze_residuals/13_cli_lambda2_visualize.py`
Linear- vs. log-clock SSM, residual (ssm_dim=3, poly=2 → λ=1.983≈2). `--no-show --save <target>`

![ssm res log](figs/fig_ssm_res_log.png)

### `analyze_n_ens/13_cli_lambda2_visualize.py`
Same for the exponent (λ≈1.24). `--ssm_dim 3 --no-show --save <target>`

![ssm exp log](figs/fig_ssm_exp_log.png)

### `SSM/res/surrogate_ssm.py`
IAAFT surrogate test (p_rank=0.02, 2.9σ). ⚠ fixed output `surrogate_ssm_fig1/2.png`

![surrogate test](figs/fig_surrogate.png)

### `attractor_scan_recurrence.py`
RQA τ scan (%DET≈0.98 ∀τ). ⚠ fixed output `recurrence_scan.png`

![recurrence scan](figs/fig_recurrence.png)

### `attractor_compare_raw_vs_ssm-linear.py`
2×2 embedding comparison: M=6 vs. M=35, raw vs. +PCA — linear-time sampled, halving-cycle colours. ⚠ fixed output `compare_raw_vs_ssm-linear.png`

![compare raw vs ssm linear](figs/compare_raw_vs_ssm-linear.png)

### `attractor_compare_raw_vs_ssm-logtime.py`
Same, log10-time sampled (continuous log10(day) colourbar). ⚠ fixed output `compare_raw_vs_ssm-logtime.png`

![compare raw vs ssm logtime](figs/compare_raw_vs_ssm-logtime.png)

---

## analyze_residuals/ (residual)
| script | purpose | clock | output |
|---|---|---|---|
| 02_cli_harmonics.py | PSD per PC | --time_mode | show only |
| 03_cli_phase.py | phase coupling 2φ1−φ2 | --time_mode | show only |
| 03b_cli_phase_all.py | phase-all, log fixed | log | --save (auto) |
| 04_cli_scaling.py | amplitude scaling | --time_mode | show only |
| 05_cli_scan.py | SSM sweep dim×poly | --time_mode | show only |
| 10_cli_ssm_learn.py | SSMLearn slave test | --time_mode | show only |
| 13_cli_lambda2_visualize.py | linear vs. log SSM (λ2) | both | --save |
| 13_2_cli_lambda2_visualize.py | same, 2nd layout (dim9) | both | --save |
| 14_cli_phase3d.py | 3D phase space | --time_mode | --save |
| 15_cli_resample_control.py | prewhitening control | both | savefig auto |

## analyze_n_ens/ (exponent)
13_cli / 13_2_cli / 14_cli / 03b analogous; additionally `--windows` (default [180]), `--log_uniform_mean/_both`.

## Root (selection with output flag)
| script | purpose | output flag |
|---|---|---|
| ssmlearn_res.py | SSMLearnPy on residuals (eigenvalues) | --out |
| ssmlearn_manifold.py | 2D SSM as 3D surface | --out |
| harmonic_test_phase.py / _n_phase.py | phase locking res/exp | --out |
| harmonic_test_backbone/_bicoherence/_slave.py | backbone/bicoherence/slave | --out |
| attractor_verify.py / _surrogates.py | kNN cross-check, Kaplan-Glass | --save |
| lpplattr02_poincare_native/_strobe.py | LPPL Poincaré | --save-prefix |
| attractor_scan_recurrence.py | RQA τ scan | ⚠ fixed |
| attractor_compare_raw_vs_ssm-{linear,logtime}.py | 2×2 comparison M=6 vs M=35 | ⚠ fixed |
| attractor_scan_m.py / _mutual.py / attractor*.py (older) | scans / older attractors | show only |

## SSM/res|n|raw (second pipeline, tau-based)
| script | purpose | output |
|---|---|---|
| SSM_res.py | full residual SSM (4 figs) | --save-prefix |
| SSM_res_free.py | pure TDE / intrinsic dim | --save-prefix |
| 13_cli_manifold.py (res+n) | SSM manifold 3D, --time_mode | show only |
| 11/12_cli (res) | embedding modes / intrinsic dim | show only |
| surrogate_ssm.py | IAAFT surrogates | ⚠ fig1/fig2 fixed |
| SSM/raw/SSM_raw.py | raw-price SSM, --log/--lin | --save-prefix |
| 02–10_cli (SSM/res) | synthetic LPPL system | show only |

## surrogates/
| script | purpose | output |
|---|---|---|
| surrogates_theiler_test.py / _n.py | Theiler tests res/exp (n_surr=999) | ⚠ fixed |
| surrogates_n_IAAFT.py (+_compute/_plot) | IAAFT exponent, --loc/--norm/--cat | show/JSON |
| recurrence_tau_scan.py | RQA τ selection | show only |
| m_scan.png | M optimisation (artifact) | — |
