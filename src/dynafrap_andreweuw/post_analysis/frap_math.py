# =============================================================================
# Mathematical FRAP model functions
# =============================================================================
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dynafrap_andreweuw.post_analysis.config import INITIAL_K, MAX_FIT_EVALUATIONS, MIN_K, TIME_COL


def one_phase_recovery(t, I_inf, k):
    """
    One-phase exponential FRAP recovery model, anchored to the known
    normalization boundary condition I(0) = 0 (the immediate post-bleach
    frame is always 0% by definition, so it is not fit as a free parameter).

    Model:

        I(t) = I_inf * (1 - exp(-k * t))

    where:

        I_inf = fitted recovery plateau (also the recovery amplitude,
                since I(0) = 0)
        k     = recovery rate constant
    """
    return I_inf * (1.0 - np.exp(-k * t))


def recovery_half_time(k):
    """
    Half-time of recovery for one-phase exponential recovery.

        t_1/2 = ln(2) / k
    """
    return np.log(2) / k


def recovery_half_intensity(I_0, A):
    """
    Intensity halfway between I_0 and I_inf.

        I_half = I_0 + 0.5 * A
    """
    return I_0 + 0.5 * A


def mobile_fraction_from_fit(I_pre, I_0, A):
    """
    Mobile fraction from fitted recovery parameters.

    The bleached amount is:

        I_pre - I_0

    The recovered amount is:

        A = I_inf - I_0

    Therefore:

        mobile_fraction = A / (I_pre - I_0)

    Returns:
        mobile_fraction, immobile_fraction
    """
    denominator = I_pre - I_0

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan, np.nan

    mobile_fraction = A / denominator
    mobile_fraction = np.clip(mobile_fraction, 0.0, 1.0)

    immobile_fraction = 1.0 - mobile_fraction

    return mobile_fraction, immobile_fraction


def initial_fit_parameters(y):
    """
    Generate initial guesses for one-phase recovery fitting.
    """
    I_inf_0 = max(y[-1] - y[0], 1e-6)
    k_0 = INITIAL_K

    return [I_inf_0, k_0]


def fit_recovery_curve(time_values, intensity_values):
    """
    Fit the one-phase recovery model to a recovery curve.

    The time axis is shifted so that the first post-bleach fitted frame is t = 0.

    Returns:
        fit_params: dict
        fit_curve: dataframe
    """
    x = np.asarray(time_values, dtype=float)
    y = np.asarray(intensity_values, dtype=float)
    print(x)
    print(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return None, None

    t_start = x.min()
    t_fit = x - t_start

    p0 = initial_fit_parameters(y)

    bounds = (
        [0, MIN_K],
        [np.inf, np.inf],
    )

    try:
        popt, _ = curve_fit(
            one_phase_recovery,
            t_fit,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=MAX_FIT_EVALUATIONS,
        )
    except RuntimeError:
        return None, None

    I_inf, k = popt

    I_0 = 0.0
    A = I_inf
    t_half = recovery_half_time(k)
    absolute_t_half = t_start + t_half
    I_half = recovery_half_intensity(I_0, A)

    t_smooth = np.linspace(t_fit.min(), t_fit.max(), 300)
    y_smooth = one_phase_recovery(t_smooth, I_inf, k)
    x_smooth = t_smooth + t_start

    fit_params = {
        "I_0": I_0,
        "I_inf": I_inf,
        "A": A,
        "k": k,
        "t_half": t_half,
        "absolute_t_half": absolute_t_half,
        "I_half": I_half,
        "t_start": t_start,
    }

    fit_curve = pd.DataFrame(
        {
            TIME_COL: x_smooth,
            "fit": y_smooth,
        }
    )

    return fit_params, fit_curve
