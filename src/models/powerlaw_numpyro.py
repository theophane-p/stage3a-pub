from typing import Optional

import numpy as np
import jax.numpy as jnp
import numpyro as npro
import numpyro.distributions as dist


def model(
        x: jnp.array,
        y: Optional[jnp.array] = None,
        y_mean: Optional[float] = 1.,
        y_std: Optional[float] = 1.,
        exponent: Optional[float] = None,
        slope: Optional[float] = None,
        intercept: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> jnp.array:

    x = jnp.maximum(x, jnp.finfo(jnp.float32).eps)
    x_mean = jnp.mean(x)
    logx = jnp.log(x)
    exponent = npro.sample(
        "exponent", 
        dist.Gamma(concentration=2., rate=1.), 
        obs=exponent)
    slope = npro.sample(
        "slope", 
        dist.LogNormal(loc=jnp.log(y_mean / x_mean), scale=3.),
        obs=slope)
    intercept = npro.sample(
        "intercept", 
        dist.Exponential(rate=3./y_mean),
        obs=intercept)
    tau = npro.sample(
        "tau", 
        dist.InverseGamma(concentration=2., rate=1./y_std), 
        obs=tau)
    with npro.plate("N", len(x)):
        y_mod = intercept + slope * jnp.power(x, exponent)
        y = npro.sample("y", dist.Normal(y_mod, tau), obs=y)
    return y


def denormalize_sample(
        sample: dict,
        x_mean: float,
        y_mean: float
) -> None:
    r"""Denormalize a sample from the model.

    Parameters
    ----------
    sample : dict
        The sample from the model.
    x_mean : float
        The mean of the x values.
    y_mean : float
        The mean of the y values.
    
    """
    sample["slope"] = sample["slope"] * y_mean / np.power(x_mean, sample["exponent"])
    sample["intercept"] = sample["intercept"] * y_mean
    sample["tau"] = sample["tau"] * y_mean