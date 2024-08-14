import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from typing import Optional, Callable

import jax
import jax.numpy as jnp
import jax.random as jrd
import numpyro as npro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, Predictive

import sys
sys.path.append("src")
from data import paths
from models import powerlaw_numpy as pwr
from models import fractional_numpy as frac
from models import powerlaw_numpyro as pwr_npro
from models import fractional_numpyro as frac_npro
from report_plots import utils

paths_dict = paths.get_default_path_dict()
dataset_a = pd.read_csv(paths_dict["dataset_a"])
out_dir = paths.get_out_dir()


def estimation_data(
        key: jax.random.PRNGKey,
        dataset: pd.DataFrame, 
        model: Callable,
        init_params: Optional[dict] = None
) -> MCMC:

    x = dataset["wind_speed"].values.copy()
    x = jnp.array(x)
    x_mean = jnp.mean(x)
    x_normalized = x / x_mean

    y = dataset["signal"].values.copy()
    y = jnp.array(y)
    y_mean = jnp.mean(y)
    y_normalized = y / y_mean

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=2000, num_chains=1)
    key, subkey = jrd.split(key)
    mcmc.run(
        subkey, 
        x=x_normalized,
        y=y_normalized,
        y_mean=jnp.mean(y_normalized), 
        y_std=jnp.std(y_normalized),
        init_params=init_params
    )
    
    return mcmc, x_mean, y_mean


def print_summary_stat_tex(res: dict) -> None:
    data_az = az.from_dict(res)
    ess = az.ess(data_az)
    print("\\hline")
    print("Variable & Median & 5\\% & 95\\% & ESS\\\\")
    print("\\hline")
    for k, v in res.items():
        arr = [np.median(v), np.quantile(v, 0.05), np.quantile(v, 0.95), ess[k].values]
        arr_str = utils.format_array1d_tex(arr)
        print(f"{k} & {arr_str} \\\\")
    print("\\hline")


def save_pwr(key: jax.random.PRNGKey) -> jax.random.PRNGKey:

    map_res = np.load(os.path.join(out_dir, "backup", "map_results.npz"))
    theta_init = pwr.ThetaClass()
    theta_init.from_array(map_res["pwr_theta_opt"])
    pwr_init_params = {
        "exponent": jnp.array(theta_init.exponent),
        "slope": jnp.array(theta_init.slope),
        "intercept": jnp.array(theta_init.intercept),
        "tau": jnp.array(theta_init.tau),
    }

    key, subkey = jrd.split(key)
    mcmc_pwr, x_mean, y_mean = estimation_data(
        subkey, 
        dataset_a, 
        pwr_npro.model,
        init_params=pwr_init_params
    )

    mcmc_pwr.transfer_states_to_host()
    posterior_samples = mcmc_pwr.get_samples(group_by_chain=False).copy()
    extra_fields = mcmc_pwr.get_extra_fields().copy()
    pwr_npro.denormalize_sample(posterior_samples, x_mean, y_mean)

    def pwr_dic(post_samples):

        x = dataset_a["wind_speed"].values
        y = dataset_a["signal"].values

        theta_mean = pwr.ThetaClass(
            exponent=np.mean(post_samples["exponent"]),
            slope=np.mean(post_samples["slope"]),
            intercept=np.mean(post_samples["intercept"]),
            tau=np.mean(post_samples["tau"])
        )
        nll_mean = pwr.nll(theta_mean, x, y)

        n_samples = post_samples["exponent"].shape[0]
        nlls = np.zeros(n_samples)

        for i in range(n_samples):
            theta = pwr.ThetaClass(
                exponent=post_samples["exponent"][i],
                slope=post_samples["slope"][i],
                intercept=post_samples["intercept"][i],
                tau=post_samples["tau"][i]
            )
            nlls[i] = pwr.nll(theta, x, y)
        mean_nll = np.mean(nlls)
        
        return 2 * mean_nll - nll_mean

    pwr_dic_value = pwr_dic(posterior_samples)

    np.savez(
        os.path.join(out_dir, "backup", "pwr_npro_estimation_data.npz"), 
        **posterior_samples, **extra_fields, dic_value=pwr_dic_value)
    data_az = az.from_dict(posterior_samples)

    plt.close("all")
    az.plot_trace(data_az)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "img", "trace_pwr.png"), dpi=300, bbox_inches="tight")

    plt.close("all")
    az.plot_pair(data_az)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "img", "pair_pwr.png"), dpi=300, bbox_inches="tight")

    plt.close("all")
    print("=" * 80)
    print("Powerlaw model")
    mcmc_pwr.print_summary()
    print_summary_stat_tex(posterior_samples)

    return pwr_dic_value


def retrieve_pwr():
    res = np.load(os.path.join(out_dir, "backup", "pwr_npro_estimation_data.npz"))
    res_dict = {k: v for k, v in res.items() if k not in ["diverging", "dic_value"]}
    print_summary_stat_tex(res_dict)


def save_frac(key: jax.random.PRNGKey) -> jax.random.PRNGKey:

    map_res = np.load(os.path.join(out_dir, "backup", "map_results.npz"))
    theta_init = frac.ThetaClass()
    theta_init.from_array(map_res["frac_theta_opt"])
    frac_init_params = {
        "exponent": jnp.array(theta_init.exponent),
        "add_const": jnp.array(theta_init.add_const),
        "mul_const": jnp.array(theta_init.mul_const),
        "x_scale": jnp.array(theta_init.x_scale),
        "tau": jnp.array(theta_init.tau),
    }

    key, subkey = jrd.split(key)
    mcmc_frac, x_mean, y_mean = estimation_data(
        subkey, 
        dataset_a, 
        frac_npro.model,
        init_params=frac_init_params
    )

    mcmc_frac.transfer_states_to_host()
    posterior_samples = mcmc_frac.get_samples(group_by_chain=False).copy()
    extra_fields = mcmc_frac.get_extra_fields().copy()
    frac_npro.denormalize_sample(posterior_samples, x_mean, y_mean)

    def frac_dic(post_samples):

        x = dataset_a["wind_speed"].values
        y = dataset_a["signal"].values

        theta_mean = frac.ThetaClass(
            exponent=np.mean(post_samples["exponent"]),
            add_const=np.mean(post_samples["add_const"]),
            mul_const=np.mean(post_samples["mul_const"]),
            x_scale=np.mean(post_samples["x_scale"]),
            tau=np.mean(post_samples["tau"])
        )
        nll_mean = frac.nll(theta_mean, x, y)

        n_samples = post_samples["exponent"].shape[0]
        nlls = np.zeros(n_samples)

        for i in range(n_samples):
            theta = frac.ThetaClass(
                exponent=post_samples["exponent"][i],
                add_const=post_samples["add_const"][i],
                mul_const=post_samples["mul_const"][i],
                x_scale=post_samples["x_scale"][i],
                tau=post_samples["tau"][i]
            )
            nlls[i] = frac.nll(theta, x, y)
        mean_nll = np.mean(nlls)
        
        return 2 * mean_nll - nll_mean

    frac_dic_value = frac_dic(posterior_samples)

    np.savez(
        os.path.join(out_dir, "backup", "frac_npro_estimation_data.npz"),
        **posterior_samples, **extra_fields, dic_value=frac_dic_value)
    data_az = az.from_dict(posterior_samples)

    plt.close("all")
    az.plot_trace(data_az)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "img", "trace_frac.png"), dpi=300, bbox_inches="tight")

    plt.close("all")
    az.plot_pair(data_az)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "img", "pair_frac.png"), dpi=300, bbox_inches="tight")

    plt.close("all")
    print("=" * 80)
    print("Fractional model")
    mcmc_frac.print_summary()
    print_summary_stat_tex(posterior_samples)

    return frac_dic_value


def retrieve_frac():
    res = np.load(os.path.join(out_dir, "backup", "frac_npro_estimation_data.npz"))
    res_dict = {k: v for k, v in res.items() if k not in ["diverging", "dic_value"]}
    print_summary_stat_tex(res_dict)


def main():
    
    pwr_dic_value = save_pwr(jax.random.PRNGKey(0))
    frac_dic_value = save_frac(jax.random.PRNGKey(0))

    print("=" * 80)
    print("DIC values: powerlaw, fractional, fractional - powerlaw")
    print(utils.format_float_tex(pwr_dic_value))
    print(utils.format_float_tex(frac_dic_value))
    print(utils.format_float_tex(frac_dic_value - pwr_dic_value))


if __name__ == "__main__":
    main()

