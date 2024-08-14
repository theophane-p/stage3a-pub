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
import utils

paths_dict = paths.get_default_path_dict()
data_df = pd.read_csv(paths_dict["dataset_a"])


def fit_powerlaw_ols(df):

    r"""Fit a power law model to the data using ordinary least squares."""

    # preparing data
    x = df["wind_speed"].values.copy()
    x = np.maximum(x, np.finfo(np.float32).eps) # avoiding log(0) in the power law
    x_mean = np.mean(x)
    x_normalized = x / x_mean
    y = df["signal"].values.copy()
    y_mean = np.mean(y)
    y_normalized = y / y_mean

    # fitting
    def objective(params):
        theta = pwr.ThetaClass()
        theta.from_array(np.concatenate([params, np.array([0.])]))
        y_pred = pwr.forward(theta, x_normalized)
        residuals = y_normalized - y_pred
        return residuals
    theta_init = pwr.get_default_theta(x_normalized, y_normalized)
    params_init = theta_init.to_array()[:-1]
    bounds = (0., np.inf)
    sol = sp.optimize.least_squares(objective, params_init, bounds=bounds, method="trf")

    # recovering parameters in true scale
    theta_opt = pwr.ThetaClass()
    theta_opt.from_array(np.concatenate([sol.x, np.array([0.])]))
    theta_opt.intercept = y_mean * theta_opt.intercept
    theta_opt.slope = y_mean * theta_opt.slope / np.power(x_mean, theta_opt.exponent)

    return theta_opt, sol


def fit_fractional_ols(df):

    r"""Fit a power law model to the data using ordinary least squares."""

    # preparing data
    x = df["wind_speed"].values.copy()
    x = np.maximum(x, np.finfo(np.float32).eps) # avoiding log(0) in the power law
    x_mean = np.mean(x)
    x_normalized = x / x_mean
    y = df["signal"].values.copy()
    y_mean = np.mean(y)
    y_normalized = y / y_mean

    # fitting
    def objective(params):
        theta = frac.ThetaClass()
        theta.from_array(np.concatenate([params, np.array([0.])]))
        y_pred = frac.forward(theta, x_normalized)
        residuals = y_normalized - y_pred
        return residuals
    theta_init = frac.get_default_theta(x_normalized, y_normalized)
    params_init = theta_init.to_array()[:-1]
    bounds = bounds = (0., np.inf)
    sol = sp.optimize.least_squares(objective, params_init, bounds=bounds, method="trf")

    # recovering parameters in true scale
    theta_opt = frac.ThetaClass()
    theta_opt.from_array(np.concatenate([sol.x, np.array([0.])]))
    theta_opt.add_const = y_mean * theta_opt.add_const
    theta_opt.mul_const = y_mean * theta_opt.mul_const
    theta_opt.x_scale = x_mean * theta_opt.x_scale

    return theta_opt, sol


def save_results():

    out_dir = paths.get_out_dir()
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "backup"), exist_ok=True)

    # preparing data
    x = data_df["wind_speed"].values.copy()
    y = data_df["signal"].values.copy()
    x_pred = np.linspace(np.min(x), np.max(x), 1000)
    pwr_theta_opt, sol_pwr = fit_powerlaw_ols(data_df)
    pwr_y_pred = pwr.forward(pwr_theta_opt, x_pred)
    frac_theta_opt, sol_frac = fit_fractional_ols(data_df)
    frac_y_pred = frac.forward(frac_theta_opt, x_pred)

    # plotting hist2d
    fig, ax = plt.subplots(figsize=(8, 5))
    hist, xedges, yedges, image = ax.hist2d(x, y, bins=(200, 100), alpha=0.8, cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1., clip=True))
    plt.colorbar(image, ax=ax, label="data points count")
    ax.plot(x_pred, pwr_y_pred, "r-", label="fit power law")
    ax.plot(x_pred, frac_y_pred, "r--", label="fit fractional")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Strain")
    ax.legend(loc="best")
    ax.grid()
    fig.savefig(os.path.join(out_dir, "img", "ols_hist2d.png"), dpi=300, bbox_inches="tight")

    # saving params
    np.savez(
        os.path.join(out_dir, "backup", "ols_results.npz"),
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
    print("final cost:", sol_pwr.cost)
    print(pwr_theta_opt)

    print("\n" + "="*80)
    print("fractional model:")
    print("initial params:", utils.format_array1d_tex(frac.get_default_theta(x, y).to_print_array()))
    print("opt params", utils.format_array1d_tex(frac_theta_opt.to_print_array()))
    print("success:", sol_frac.success)
    print("message:", sol_frac.message)
    print("final cost:", sol_frac.cost)
    print(frac_theta_opt)


if __name__ == "__main__":

    save_results()
