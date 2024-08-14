from typing import Any, Optional, Callable
import os
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append("src")
from data import paths
from models import powerlaw_numpy as pwr
from models import powerlaw_svc_numpy as pwr_svc
from report_plots import utils

paths_dict = paths.get_default_path_dict()
data_df = pd.read_csv(paths_dict["dataset_a"], parse_dates=["time"])


def prepare_x_y_svc(df:pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    df = df.copy()
    loc_np = np.sort(np.array(df["channel"].unique()))
    df_multiindex = df.set_index(["time", "channel"])
    ds = xr.Dataset.from_dataframe(df_multiindex[["signal", "wind_speed"]])
    x_np = ds.wind_speed.values
    y_np = ds.signal.values
    return x_np, y_np, loc_np


def fit_svc(df:pd.DataFrame) -> tuple[pwr_svc.ThetaClass, Any]:

    x, y, loc = prepare_x_y_svc(df)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x = x / x_mean
    y = y / y_mean
    theta_init = pwr_svc.get_default_theta(x=x, y=y)
    bounds = theta_init.get_bounds()
    prior = pwr_svc.get_default_prior(x=x, y=y)

    def to_optimize(theta_array: np.ndarray) -> float:
        theta = pwr_svc.ThetaClass()
        theta.from_array(theta_array)
        return pwr_svc.nlp(
            theta=theta,
            locations=loc,
            x=x,
            y=y,
            prior=prior
        )

    sol = sp.optimize.minimize(
        fun=to_optimize,
        x0=theta_init.to_array(),
        method="Nelder-Mead",
        tol=100.,
        options={"maxiter": 30000},
        bounds=bounds,
    )

    theta_opt = pwr_svc.ThetaClass()
    theta_opt.from_array(sol.x)
    theta_opt = theta_opt.get_denormalized(x_mean=x_mean, y_mean=y_mean)
    x = x * x_mean
    y = y * y_mean
    prediction_fun = pwr_svc.get_prediction_function(
        theta=theta_opt,
        locations=loc,
        x=x,
        y=y,
    )
    return theta_opt, sol, prediction_fun


def get_fit_df(df:pd.DataFrame, prediction_fun:Callable) -> pd.DataFrame:

    fit_df = df.copy()
    fit_df["pred"] = fit_df.apply(
        func= lambda row: prediction_fun(row["wind_speed"], row["channel"]),
        axis=1
    )
    fit_df["residual"] = fit_df["signal"] - fit_df["pred"]
    return fit_df


def save_results():

    out_dir = paths.get_out_dir()

    kakern_df = data_df[data_df["weather_source"] == "kakern"]
    kakern_theta, kakern_sol, kakern_pfun = fit_svc(kakern_df)
    kakern_fit_df = get_fit_df(kakern_df, kakern_pfun)

    leknes_df = data_df[data_df["weather_source"] == "leknes"]
    leknes_theta, leknes_sol, leknes_pfun = fit_svc(leknes_df)
    leknes_fit_df = get_fit_df(leknes_df, leknes_pfun)

    svc_fit_df = pd.concat([kakern_fit_df, leknes_fit_df])

    map_res = np.load(os.path.join(out_dir, "backup", "map_results.npz"))
    theta_pwr = pwr.ThetaClass()
    theta_pwr.from_array(map_res["pwr_theta_opt"])
    pwr_fit_df = data_df.copy()
    pwr_fit_df["pred"] = pwr.forward(theta_pwr, pwr_fit_df["wind_speed"].values)
    pwr_fit_df["residual"] = pwr_fit_df["signal"] - pwr_fit_df["pred"]

    print("=" * 80)
    print("Mean absolute residuals (simple power law, power law with SVC):")
    for x in [pwr_fit_df["residual"], svc_fit_df["residual"]]:
        print(utils.format_float_tex(np.mean(np.abs(x))))

    print("=" * 80)
    print("standard deviation of das signal, power law with SVC RMS of residuals:")
    for x in [
        svc_fit_df["signal"].std(), 
        np.sqrt(np.mean(np.square(svc_fit_df["residual"].values)))]:
        print(utils.format_float_tex(x)) 

    def plot_res_v_speed(
            df:pd.DataFrame, 
            ax:Optional[matplotlib.axes.Axes]=None,
            title:Optional[str]=None
    ) -> matplotlib.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        pd_plot_kwargs_a = dict(
            s=0.7,
            alpha=0.2, 
            grid=True
        )
        ax.set_title(title)
        df.plot.scatter(x="wind_speed", y="residual", ax=ax, **pd_plot_kwargs_a)
        return ax

    plt.close("all")
    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    plot_res_v_speed(svc_fit_df, ax1, title="Power law with SVC")
    plot_res_v_speed(pwr_fit_df, ax2, title="Simple power law")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "img", "svc_residuals_vs_wind_speed.png"), dpi=300, bbox_inches="tight")

    def plot_res_v_channel(
            df:pd.DataFrame, 
            ax:Optional[matplotlib.axes.Axes]=None,
            title:Optional[str]=None
    ) -> matplotlib.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        pd_plot_kwargs_a = dict(
            s=0.7,
            alpha=0.2, 
            grid=True
        )
        ax.set_title(title)
        df[df["channel"] >= 1000].plot.scatter(x="channel", y="residual", ax=ax, **pd_plot_kwargs_a)
        return ax

    plt.close("all")
    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharey=ax1)
    plot_res_v_channel(svc_fit_df, ax1, title="Power law with SVC")
    plot_res_v_channel(pwr_fit_df, ax2, title="Simple power law")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "img", "svc_residuals_vs_channel.png"), dpi=300, bbox_inches="tight")

    pd_plot_kwargs_a = dict(
        s=0.7,
        alpha=0.1, 
        grid=True
    )
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 4))
    def filter(df, station):
        return (df["time"] >= pd.Timestamp("2024-01-25")) \
            & (df["time"] <= pd.Timestamp("2024-01-28")) \
            & (df["weather_source"] == station)
    svc_fit_df[filter(svc_fit_df, "kakern")].plot.scatter(x="time", y="residual", ax=ax, color="blue", label="Kåkern (blue)", **pd_plot_kwargs_a)
    svc_fit_df[filter(svc_fit_df, "leknes")].plot.scatter(x="time", y="residual", ax=ax, color="red", label="Leknes (red)", **pd_plot_kwargs_a)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "img", "svc_residuals_vs_time.png"), dpi=300, bbox_inches="tight")

    np.savez(
        os.path.join(out_dir, "backup", "svc_results.npz"),
        kakern_theta=kakern_theta.to_array(),
        kakern_sol=kakern_sol,
        leknes_theta=leknes_theta.to_array(),
        leknes_sol=leknes_sol,
    )


if __name__ == "__main__":
    save_results()