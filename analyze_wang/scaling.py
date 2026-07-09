"""Amplitude-scaling diagnostics for Wang harmonics."""

import numpy as np


def fit_zero_intercept(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = float(np.dot(x, x))
    slope = 0.0 if denom <= 1e-30 else float(np.dot(x, y) / denom)
    y_hat = slope * x
    resid = y - y_hat
    sse = float(np.dot(resid, resid))
    sst0 = float(np.dot(y, y))
    r2_zero = 1.0 - sse / max(sst0, 1e-30)
    return {
        'slope': slope,
        'y_hat': y_hat,
        'sse': sse,
        'r2_zero': float(r2_zero),
    }


def scaling_report(main_envelope, harm_envelope, min_main_quantile=0.2):
    main_env = np.asarray(main_envelope, dtype=float)
    harm_env = np.asarray(harm_envelope, dtype=float)
    x = main_env ** 2
    y = harm_env

    threshold = float(np.quantile(main_env, min_main_quantile))
    mask = main_env >= threshold
    x_m = x[mask]
    y_m = y[mask]

    fit = fit_zero_intercept(x_m, y_m)
    corr = float(np.corrcoef(x_m, y_m)[0, 1]) if len(x_m) >= 2 else float('nan')
    main_mean = float(np.mean(main_env))
    harm_mean = float(np.mean(harm_env))
    main_std = float(np.std(main_env))
    harm_std = float(np.std(harm_env))
    main_cv = main_std / max(abs(main_mean), 1e-30)
    harm_cv = harm_std / max(abs(harm_mean), 1e-30)
    x_center = x_m - float(np.median(x_m))
    y_center = y_m - float(np.median(y_m))
    corr_centered = float(np.corrcoef(x_center, y_center)[0, 1]) if len(x_m) >= 2 else float('nan')
    scaling_identifiable = bool(main_cv >= 0.05)

    return {
        'mask': mask,
        'threshold_main': threshold,
        'x_main_sq': x,
        'y_harm': y,
        'y_hat_full': fit['slope'] * x,
        'x_masked': x_m,
        'y_masked': y_m,
        'slope_zero': fit['slope'],
        'r2_zero': fit['r2_zero'],
        'corr': corr,
        'corr_centered': corr_centered,
        'main_mean': main_mean,
        'harm_mean': harm_mean,
        'main_std': main_std,
        'harm_std': harm_std,
        'main_cv': float(main_cv),
        'harm_cv': float(harm_cv),
        'scaling_identifiable': scaling_identifiable,
        'y_hat_masked': fit['y_hat'],
    }
