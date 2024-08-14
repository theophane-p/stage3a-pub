import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join("src"))
from data import paths
from models import powerlaw_numpy as pwr
from models import fractional_numpy as frac
from report_plots import utils

paths_dict = paths.get_default_path_dict()
data_df = pd.read_csv(paths_dict["dataset_a"])

def fit_powerlaw_map(df):

    r"""Fit a power law model to the dataset using maximum a posteriori estimation."""

    # preparing data
    x = df["wind_speed"].values.copy()
    x_mean = np.mean(x)
    x_normalized = x / x_mean
    y = df["signal"].values.copy()
    y_mean = np.mean(y)
    y_normalized = y / y_mean

    prior = pwr.get_default_prior(x_normalized, y_normalized)
    def to_optimize(params : np.ndarray) -> float:
        theta = pwr.ThetaClass()
        theta.from_array(params)
        return pwr.nlp(theta, x_normalized, y_normalized, prior)
    theta_init = pwr.get_default_theta(x_normalized, y_normalized)
    bounds = theta_init.get_bounds()
    sol = sp.optimize.minimize(
            fun=to_optimize,
            x0=theta_init.to_array(),
            method="Nelder-Mead",
            tol=1e-6,
            options={"maxiter": 30000},
            bounds=bounds,
        )
    params_opt = sol.x
    theta_opt = pwr.ThetaClass()
    theta_opt.from_array(params_opt)
    theta_opt = theta_opt.get_denormalized(x_mean, y_mean)

    return theta_opt, sol


def fit_fractional_map(df):

    r"""Fit the fractional model to the dataset using maximum a posteriori estimation."""

    # preparing data
    x = df["wind_speed"].values.copy()
    x_mean = np.mean(x)
    x_normalized = x / x_mean
    y = df["signal"].values.copy()
    y_mean = np.mean(y)
    y_normalized = y / y_mean

    prior = frac.get_default_prior(x_normalized, y_normalized)
    def to_optimize(params : np.ndarray) -> float:
        theta = frac.ThetaClass()
        theta.from_array(params)
        return frac.nlp(theta, x_normalized, y_normalized, prior)
    theta_init = frac.get_default_theta(x_normalized, y_normalized)
    bounds = theta_init.get_bounds()
    sol = sp.optimize.minimize(
            fun=to_optimize,
            x0=theta_init.to_array(),
            method="Nelder-Mead",
            tol=1e-6,
            options={"maxiter": 30000},
            bounds=bounds,
        )
    params_opt = sol.x
    theta_opt = frac.ThetaClass()
    theta_opt.from_array(params_opt)
    theta_opt = theta_opt.get_denormalized(x_mean, y_mean)
    
    return theta_opt, sol


def save_results():

    out_dir = paths.get_out_dir()
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "backup"), exist_ok=True)

    # confidence interval
    confidence = 0.95
    low, high = sp.stats.norm.interval(confidence, loc=0, scale=1)

    # preparing data
    x = data_df["wind_speed"].values.copy()
    y = data_df["signal"].values.copy()
    x_pred = np.linspace(np.min(x), np.max(x), 1000)
    pwr_theta_opt, sol_pwr = fit_powerlaw_map(data_df)
    pwr_y_pred = pwr.forward(pwr_theta_opt, x_pred)
    pwr_y_pred_low = pwr_y_pred + low * pwr_theta_opt.tau
    pwr_y_pred_high = pwr_y_pred + high * pwr_theta_opt.tau
    frac_theta_opt, sol_frac = fit_fractional_map(data_df)
    frac_y_pred = frac.forward(frac_theta_opt, x_pred)
    frac_y_pred_low = frac_y_pred + low * frac_theta_opt.tau
    frac_y_pred_high = frac_y_pred + high * frac_theta_opt.tau 

    # plotting hist2d
    fig, ax = plt.subplots(figsize=(8, 5))
    hist, xedges, yedges, image = ax.hist2d(x, y, bins=(200, 100), alpha=0.8, cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1., clip=True))
    plt.colorbar(image, ax=ax, label="data points count")
    ax.plot(x_pred, pwr_y_pred, "r-", label="fit power law")
    confidence_kwargs = dict(color="white", linewidth=0.8)
    ax.plot(x_pred, pwr_y_pred_low, "-", **confidence_kwargs)
    ax.plot(x_pred, pwr_y_pred_high, "-", **confidence_kwargs)
    ax.plot(x_pred, frac_y_pred, "r--", label="fit fractional")
    ax.plot(x_pred, frac_y_pred_low, "--", **confidence_kwargs)
    ax.plot(x_pred, frac_y_pred_high, "--", **confidence_kwargs)
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Strain")
    ax.legend(loc="best")
    ax.grid()
    fig.savefig(os.path.join(out_dir, "img", "map_hist2d.png"), dpi=300, bbox_inches="tight")

    # saving params
    np.savez(
        os.path.join(out_dir, "backup", "map_results.npz"),
        pwr_theta_opt=pwr_theta_opt.to_array(), 
        sol_pwr=sol_pwr, 
        frac_theta_opt=frac_theta_opt.to_array(), 
        sol_frac=sol_frac
    )

    print("\n" + "="*80)
    print("Power law model:")
    print("initial params:", utils.format_array1d_tex(pwr.get_default_theta(x, y).to_print_array()))
    print("opt params", utils.format_array1d_tex(pwr_theta_opt.to_print_array()))
    print("success:", sol_pwr.success)
    print("message:", sol_pwr.message)
    print("final cost:", sol_pwr.fun)
    print(pwr_theta_opt)

    print("\n" + "="*80)
    print("fractional model:")
    print("initial params:", utils.format_array1d_tex(frac.get_default_theta(x, y).to_print_array()))
    print("opt params", utils.format_array1d_tex(frac_theta_opt.to_print_array()))
    print("success:", sol_frac.success)
    print("message:", sol_frac.message)
    print("final cost:", sol_frac.fun)
    print(frac_theta_opt)


if __name__ == "__main__":
    save_results()