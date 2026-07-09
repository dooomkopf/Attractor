#!/usr/bin/env python3
"""
Fit parametric distributions to BTC log-residuals r (days >= 1164).

Goal: decide whether r is well-described by a single parametric family or
needs a 2-component mixture (Akku-Phase + Bull-Phase).
"""

import os
import sys
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

ATTRACTOR_DIR = "/home/hz/Data/Attractor"
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)
from ssmlearn_res import read_btc_data  # noqa: E402

PERCENTILE = 0.01
START_IDX = 1164


def get_residuals():
    days, prices, _ = read_btc_data(os.path.join(ATTRACTOR_DIR, "ziel.csv"))
    X = sm.add_constant(np.log(days))
    fit = QuantReg(np.log(prices), X).fit(q=PERCENTILE).predict(X)
    res = np.log(prices / np.exp(fit))
    return res[days >= START_IDX]


def aic_bic(loglik, k, n):
    return 2 * k - 2 * loglik, k * np.log(n) - 2 * loglik


def fit_scipy(name, dist, r, floc=None, fscale=None, **fkw):
    """Fit a scipy continuous distribution and return metrics."""
    kwargs = dict(fkw)
    if floc is not None:
        kwargs["floc"] = floc
    if fscale is not None:
        kwargs["fscale"] = fscale
    try:
        params = dist.fit(r, **kwargs)
    except Exception as e:
        return {"name": name, "error": str(e)}
    # number of free parameters = total - fixed
    k_total = len(params)
    k_fixed = sum(1 for v in (floc, fscale) if v is not None)
    k_free = k_total - k_fixed
    loglik = np.sum(dist.logpdf(r, *params))
    n = len(r)
    aic, bic = aic_bic(loglik, k_free, n)
    ks_stat, ks_p = stats.kstest(r, dist.cdf, args=params)
    # Anderson-Darling A^2 via empirical formula on transformed sample
    cdf_vals = np.sort(dist.cdf(r, *params))
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
    nn = len(cdf_vals)
    i = np.arange(1, nn + 1)
    a2 = -nn - np.mean((2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    return {
        "name": name,
        "params": params,
        "k": k_free,
        "loglik": loglik,
        "AIC": aic,
        "BIC": bic,
        "KS": ks_stat,
        "KS_p": ks_p,
        "A2": a2,
    }


def fit_gmm(r, n_components=2):
    gm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        n_init=10, random_state=0, max_iter=500,
    )
    gm.fit(r.reshape(-1, 1))
    n = len(r)
    # parameters: (means + variances + weights-1) for diagonal/full 1D
    k_free = n_components + n_components + (n_components - 1)
    loglik = gm.score(r.reshape(-1, 1)) * n
    aic = 2 * k_free - 2 * loglik
    bic = k_free * np.log(n) - 2 * loglik

    # GMM cdf for KS / A2
    means = gm.means_.flatten()
    stds = np.sqrt(gm.covariances_.flatten())
    weights = gm.weights_.flatten()

    def gmm_cdf(x):
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=float)
        for w, m, s in zip(weights, means, stds):
            out += w * stats.norm.cdf(x, loc=m, scale=s)
        return out

    ks_stat, ks_p = stats.kstest(r, gmm_cdf)
    cdf_vals = np.sort(gmm_cdf(r))
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)
    nn = len(cdf_vals)
    i = np.arange(1, nn + 1)
    a2 = -nn - np.mean((2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    return {
        "name": f"GMM(k={n_components})",
        "params": {"means": means, "stds": stds, "weights": weights},
        "k": k_free,
        "loglik": loglik,
        "AIC": aic,
        "BIC": bic,
        "KS": ks_stat,
        "KS_p": ks_p,
        "A2": a2,
    }


def qq_diagnostics(r, dist, params, label, n_quantiles=99):
    qs = np.linspace(0.01, 0.99, n_quantiles)
    emp = np.quantile(r, qs)
    theo = dist.ppf(qs, *params)
    resid = emp - theo
    print(f"\n[Q-Q] {label}: max |emp-theo| = {np.max(np.abs(resid)):.4f}")
    bins = [(0.05, 0.30, "lower body"), (0.30, 0.70, "central body"), (0.70, 0.95, "upper body")]
    for lo, hi, lbl in bins:
        mask = (qs >= lo) & (qs <= hi)
        seg = resid[mask]
        print(f"   {lbl:>14s} (q={lo:.2f}-{hi:.2f}): mean dev = {np.mean(seg):+.4f}, max |dev| = {np.max(np.abs(seg)):.4f}")
    return resid, qs, theo, emp


def main():
    r = get_residuals()
    n = len(r)
    print(f"N = {n}")
    print(f"mean={r.mean():.4f}  std={r.std():.4f}  median={np.median(r):.4f}")
    print(f"min={r.min():.4f}  max={r.max():.4f}  skew={stats.skew(r):.4f}  exkurt={stats.kurtosis(r):.4f}")
    print()

    # All candidates need POSITIVE support (except normal/skewnorm). Shift if needed.
    eps = 1e-3
    shift = r.min() - eps   # so r - shift > 0
    rp = r - shift
    print(f"Shift for positive-support fits: shift = r.min() - eps = {shift:.6f}")
    print(f"After shift: min={rp.min():.6f}, max={rp.max():.6f}")
    print()

    results = []

    # ---- Candidates ----
    # 1) Lognormal: requires positive support -> on rp, with floc=0 (2-param lognormal)
    results.append(fit_scipy("Lognormal(shift)", stats.lognorm, rp, floc=0))
    # 2) Gamma: positive support -> on rp, floc=0
    results.append(fit_scipy("Gamma(shift)", stats.gamma, rp, floc=0))
    # 3) Weibull (min): positive support
    results.append(fit_scipy("Weibull(shift)", stats.weibull_min, rp, floc=0))
    # 4) Generalized Gamma: positive support, 3 shape/scale params
    results.append(fit_scipy("GenGamma(shift)", stats.gengamma, rp, floc=0))
    # 5) Inverse Gaussian: positive support
    results.append(fit_scipy("InvGauss(shift)", stats.invgauss, rp, floc=0))
    # 6) Beta on rp scaled to [0,1] (need finite max). Use fscale=rp.max()*1.0001.
    scale_beta = rp.max() * 1.0001
    results.append(fit_scipy("Beta(shift,scale)", stats.beta, rp, floc=0, fscale=scale_beta))
    # 7) Stretched exponential = Weibull = already covered, but report exponweib for completeness
    results.append(fit_scipy("ExponWeib(shift)", stats.exponweib, rp, floc=0))
    # 8) Skew-normal as a single-component "asymmetric Gaussian" benchmark on raw r
    results.append(fit_scipy("SkewNorm(raw)", stats.skewnorm, r))
    # 9) Plain Normal as baseline (worst expected)
    results.append(fit_scipy("Normal(raw)", stats.norm, r))

    # GMM 2 / 3 components on raw r
    results.append(fit_gmm(r, n_components=2))
    results.append(fit_gmm(r, n_components=3))

    # ---- Print table ----
    print("=" * 95)
    print(f"{'distribution':<22s} {'k':>3s} {'loglik':>11s} {'AIC':>11s} {'BIC':>11s} {'KS':>8s} {'KS_p':>8s} {'A2':>8s}")
    print("-" * 95)
    valid = [r_ for r_ in results if "error" not in r_]
    valid.sort(key=lambda d: d["BIC"])
    for d in valid:
        print(f"{d['name']:<22s} {d['k']:>3d} {d['loglik']:>11.2f} {d['AIC']:>11.2f} {d['BIC']:>11.2f} "
              f"{d['KS']:>8.4f} {d['KS_p']:>8.4g} {d['A2']:>8.3f}")
    for d in results:
        if "error" in d:
            print(f"  [SKIP] {d['name']}: {d['error']}")
    print("=" * 95)

    # ---- Q-Q diagnostics for best single-distribution and best overall ----
    single = [d for d in valid if not d["name"].startswith("GMM")]
    best_single = single[0]
    best_overall = valid[0]
    print(f"\nBest single-distribution by BIC: {best_single['name']}  (BIC={best_single['BIC']:.2f})")
    print(f"Best overall by BIC:            {best_overall['name']}  (BIC={best_overall['BIC']:.2f})")
    delta_bic = best_single["BIC"] - best_overall["BIC"]
    print(f"Delta BIC (single - best)     : {delta_bic:+.2f}")

    # Map best single name -> scipy dist for Q-Q
    name_to_dist = {
        "Lognormal(shift)": (stats.lognorm, True),
        "Gamma(shift)": (stats.gamma, True),
        "Weibull(shift)": (stats.weibull_min, True),
        "GenGamma(shift)": (stats.gengamma, True),
        "InvGauss(shift)": (stats.invgauss, True),
        "Beta(shift,scale)": (stats.beta, True),
        "ExponWeib(shift)": (stats.exponweib, True),
        "SkewNorm(raw)": (stats.skewnorm, False),
        "Normal(raw)": (stats.norm, False),
    }
    dist, is_shifted = name_to_dist[best_single["name"]]
    if is_shifted:
        qq_diagnostics(rp, dist, best_single["params"], best_single["name"])
    else:
        qq_diagnostics(r, dist, best_single["params"], best_single["name"])

    # GMM details
    gmm2 = next(d for d in results if d["name"] == "GMM(k=2)")
    print(f"\nGMM(k=2) components:")
    for w, m, s in sorted(zip(gmm2["params"]["weights"], gmm2["params"]["means"], gmm2["params"]["stds"]),
                         key=lambda t: t[1]):
        print(f"   w={w:.3f}  mean={m:+.3f}  std={s:.3f}")

    # Recommendation
    print("\n" + "=" * 95)
    if delta_bic > 10:
        rec = f"GMM(k=2) clearly preferred (Delta BIC = {delta_bic:.1f} > 10)."
    elif delta_bic > 6:
        rec = f"GMM(k=2) strongly preferred (Delta BIC = {delta_bic:.1f})."
    elif delta_bic > 2:
        rec = f"GMM(k=2) moderately preferred (Delta BIC = {delta_bic:.1f})."
    else:
        rec = f"Single distribution {best_single['name']} sufficient (Delta BIC = {delta_bic:.1f})."
    print("RECOMMENDATION:", rec)
    print("=" * 95)


if __name__ == "__main__":
    main()
