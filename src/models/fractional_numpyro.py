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
        add_const: Optional[float] = None,
        mul_const: Optional[float] = None,
        x_scale: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> jnp.array:

    x = jnp.maximum(x, jnp.finfo(jnp.float32).eps)
    x_mean = jnp.mean(x)
    logx = jnp.log(x)
    exponent = npro.sample(
        "exponent", 
        dist.Gamma(concentration=2., rate=0.5), 
        obs=exponent)
    add_const = npro.sample(
        "add_const", 
        dist.Exponential(rate=3./y_mean),
        obs=add_const)
    mul_const = npro.sample(
        "mul_const", 
        dist.LogNormal(loc=jnp.log(y_mean / x_mean), scale=3.),
        obs=mul_const)
    x_scale = npro.sample(
        "x_scale", 
        dist.LogNormal(loc=jnp.mean(logx), scale=2.*jnp.std(logx)),
        obs=x_scale)
    tau = npro.sample(
        "tau", 
        dist.InverseGamma(concentration=2., rate=1./y_std), 
        obs=tau)
    with npro.plate("N", len(x)):
        x_b = jnp.power(x / x_scale, exponent)
        y_mod = add_const + mul_const * x_b / (1. + x_b)
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
    sample["add_const"] = sample["add_const"] * y_mean
    sample["mul_const"] = sample["mul_const"] * y_mean
    sample["x_scale"] = sample["x_scale"] * x_mean
    sample["tau"] = sample["tau"] * y_mean