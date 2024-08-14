import numpy as np
import scipy as sp
import scipy.stats as sps
from typing import Optional


"""This module contains the implementation of a power-law model with spatially varying parameters. 

Extended summary
----------------
The model used here is: 

.. math:: y_{n,t} = \beta_{0,n} + \beta_{1,n} x_{t,n}^{\alpha} + \epsilon_{t,n}

where :math:`y_{t,n}` (`y` in the code) is the DAS signal at location :math:`n` and time point :math:`t`,
:math:`x_{t,n}` (`x` in the code) is the wind speed at location :math:`n` and time point :math:`t`,
:math:`\alpha` (`theta.exponent` in the code) is the exponent in the power-law relationship between wind speed and DAS signal,
:math:`\beta_{0,n}, \beta{1,n}` are hereafter referred to as latent parameters of the model.
They are spatial Gaussian processes with prior mean :math:`\mu_0, \mu_1` and prior covariance
:math:`R_{\theta}(d_{n,n'})`, where :math:`d_{n,n'} = |n - n'|`, and :math:`R_{\theta}` is the
Matern 3/2 covariance function with length scale `\theta.length_scale` and scale `\theta.sigma_0, \theta.sigma_1`.
The noise term :math:`\epsilon_{t,n}` has a Gaussian white noise prior with variance `\theta.tau^2`.

Prior parameters are stored in a `ThetaClass` object, and the model is implemented as a set of functions.
This module enables the computation of the posterior negative log-likelihood of the model given data 
up to an additive constant w.r.t. to `theta`, 
learning the posterior mean and covariance of the latent parameters :math:`\beta_{0,n}, \beta{1,n}` on the way. 
It is intended to be used in a Bayesian optimization setting, where the negative log-likelihood is minimized w.r.t. `theta`.

We use the following conventions.
    N : int
        The number of locations.
    T : int
        The number of time points.
    x : np.ndarray, shape=(T, N)
        the wind speed at each location and time point.
    x_pow : np.ndarray, shape=(T, N)
        the wind speed raised at power theta.exponent at each location and time point.
    y : np.ndarray, shape=(T, N)
        the DAS signal at each location and time point.
    theta : ThetaClass
        the parameters of the model. See `ThetaClass` for details.
    mu : np.ndarray, shape=(2*N,)
        the prior mean of latent parameters (beta) of the model, ie givent the prior parameters theta but not the data.
    Sigma : np.ndarray, shape=(2*N,2*N)
        the prior covariance of latent parameters (beta) of the model, ie givent the prior parameters theta but not the data.
    mu_post : np.ndarray, shape=(2*N,)
        the learnt mean of latent parameters (beta) of the model, ie given the prior parameters theta and the data.
    Sigma_post : np.ndarray, shape=(2*N,2*N)
        the learnt covariance of latent parameters (beta) of the model, ie given the prior parameters theta and the data.
    <var>_inv
        the inverse of <var>, in the sense of matrix inversion.
    <var>_sq
        the square of <var>, in the sense of element-wise squaring.
"""


class ThetaClass:

    r"""Data class for the parameters of the model.
    
    Parameters
    ----------
    exponent : float
        The exponent in the power-law relationship between wind speed and DAS signal.
    mu_0 : float
        The prior mean of the first latent Gaussian process.
    mu_1 : float
        The prior mean of the second latent Gaussian process.
    sigma_0 : float
        The scale for prior variance of the first latent Gaussian process.
    sigma_1 : float
        The scale for prior variance of the second latent Gaussian process.
    length_scale : float
        The length scale of the prior covariance function.
    tau : float
        The scale for the noise variance.
    rho : float
        The prior correlation between the two Gaussian processes.
    """

    def __init__(
            self,
            exponent: Optional[float] = None,
            mu_0: Optional[float] = None,
            mu_1: Optional[float] = None,
            sigma_0: Optional[float] = None,
            sigma_1: Optional[float] = None,
            length_scale: Optional[float] = None,
            tau: Optional[float] = None,
            rho: Optional[float] = None
    ):
        self.exponent = exponent
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.length_scale = length_scale
        self.tau = tau
        self.rho = rho
    
    def __repr__(self) -> str:
        return f"powerlaw_svc_numpy ThetaClass(exponent={self.exponent}:" \
            + f"    mu_0={self.mu_0}\n" \
            + f"    mu_1={self.mu_1}\n" \
            + f"    sigma_0={self.sigma_0}\n" \
            + f"    sigma_1={self.sigma_1}\n" \
            + f"    length_scale={self.length_scale}\n" \
            + f"    tau={self.tau}\n" \
            + f"    rho={self.rho})"

    def from_array(self, arr: np.ndarray):
        self.exponent = arr[0]
        self.mu_0 = arr[1]
        self.mu_1 = arr[2]
        self.sigma_0 = arr[3]
        self.sigma_1 = arr[4]
        self.length_scale = arr[5]
        self.tau = arr[6]
        self.rho = arr[7]
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.exponent,
            self.mu_0,
            self.mu_1,
            self.sigma_0,
            self.sigma_1,
            self.length_scale,
            self.tau,
            self.rho
        ])
    
    def get_denormalized(
            self, 
            x_mean: float, 
            y_mean: float
    ) -> 'ThetaClass':
        
        x_cst = np.power(x_mean, self.exponent)
        return ThetaClass(
            exponent= self.exponent,
            mu_0= y_mean * self.mu_0,
            mu_1= y_mean * self.mu_1 / x_cst,
            sigma_0= y_mean * self.sigma_0,
            sigma_1= y_mean * self.sigma_1 / x_cst,
            length_scale= self.length_scale,
            tau= y_mean * self.tau,
            rho= self.rho
        )

    @staticmethod
    def get_bounds() -> list[tuple[Optional[float], Optional[float]]]:

        return [
            (0., None), # exponent
            (0., None), # mu_0
            (0., None), # mu_1
            (0., None), # sigma_0
            (0., None), # sigma_1
            (0., None), # length_scale
            (0., None), # tau
            (-1., 1.)    # rho
        ]
    

def get_default_theta(
        x: np.ndarray,
        y: np.ndarray,
) -> ThetaClass:
    
    r"""Compute the default parameters of the model given data.

    Parameters
    ----------
    x : np.ndarray, shape=(T, N)
        The wind speed at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return ThetaClass(
        exponent=1.0,
        mu_0=0.0,
        mu_1= y_mean / x_mean,
        sigma_0=y_std/2.0,
        sigma_1=y_std / x_mean,
        length_scale=4.0, 
        tau=y_std,
        rho=0.0
    )


class PriorClass:

    r"""Class for the prior distribution of the parameters of the model.

    Parameters are assumed to be independent, 
    and for each parameter, a prior continuous distribution from scipy stats is specified.
    
    Parameters
    ----------
    exponent : 
        The prior distribution of theta.exponent.
    mu_0 :
        The prior distribution of theta.mu_0.
    mu_1 :
        The prior distribution of theta.mu_1.
    sigma_0 :
        The prior distribution of theta.sigma_0.
    sigma_1 :
        The prior distribution of theta.sigma_1.
    length_scale :
        The prior distribution of theta.length_scale.
    tau :
        The prior distribution of theta.tau.
    rho :
        The prior distribution of theta.rho.
    """

    def __init__(
            self,
            exponent: Optional[sp.stats.rv_continuous] = None,
            mu_0: Optional[sp.stats.rv_continuous] = None,
            mu_1: Optional[sp.stats.rv_continuous] = None,
            sigma_0: Optional[sp.stats.rv_continuous] = None,
            sigma_1: Optional[sp.stats.rv_continuous] = None,
            length_scale: Optional[sp.stats.rv_continuous] = None,
            tau: Optional[sp.stats.rv_continuous] = None,
            rho: Optional[sp.stats.rv_continuous] = None
    ) -> None:
            
            self.exponent = exponent
            self.mu_0 = mu_0
            self.mu_1 = mu_1
            self.sigma_0 = sigma_0
            self.sigma_1 = sigma_1
            self.length_scale = length_scale
            self.tau = tau
            self.rho = rho
    
    def __repr__(self) -> str:

        return f"powerlaw_svc_numpy PriorClass object at {hex(id(self))}:\n" \
            + f"    exponent={self.exponent}\n" \
            + f"    mu_0={self.mu_0}\n" \
            + f"    mu_1={self.mu_1}\n" \
            + f"    sigma_0={self.sigma_0}\n" \
            + f"    sigma_1={self.sigma_1}\n" \
            + f"    length_scale={self.length_scale}\n" \
            + f"    tau={self.tau}\n" \
            + f"    rho={self.rho}"
    
    @staticmethod
    def _get_logpdf_or_zero(
        dist: Optional[sp.stats.rv_continuous],
        value: float
    ) -> float:
            
        return dist.logpdf(value) if dist is not None else 0.
    
    def logpdf(
            self,
            theta: ThetaClass
    ) -> float:
        
        return np.sum([
            self._get_logpdf_or_zero(self.exponent, theta.exponent),
            self._get_logpdf_or_zero(self.mu_0, theta.mu_0),
            self._get_logpdf_or_zero(self.mu_1, theta.mu_1),
            self._get_logpdf_or_zero(self.sigma_0, theta.sigma_0),
            self._get_logpdf_or_zero(self.sigma_1, theta.sigma_1),
            self._get_logpdf_or_zero(self.length_scale, theta.length_scale),
            self._get_logpdf_or_zero(self.tau, theta.tau),
            self._get_logpdf_or_zero(self.rho, theta.rho)
        ])
    

def get_default_prior(
        x: np.ndarray,
        y: np.ndarray
) -> PriorClass:
    
    r"""Compute the default prior distribution of the parameters of the model given data.

    Parameters
    ----------
    x : np.ndarray, shape=(T, N)
        The wind speed at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return PriorClass(
        exponent=sps.gamma(a=2., scale=1.),
        mu_0=sps.expon(scale=y_mean/3),
        mu_1=sps.lognorm(s=3., scale=y_mean/x_mean),
        sigma_0=sps.expon(scale=y_std/3),
        sigma_1=sps.expon(scale=y_std/x_mean),
        length_scale=sps.lognorm(s=3., scale=4.0),
        tau=sps.invgamma(a=2., scale=y_std),
        rho=sps.uniform(loc=-1.0, scale=2.0)
    )


def compute_mu(
        theta: ThetaClass,
        N: int
) -> np.ndarray:

    r"""Compute the prior mean of the latent parameters.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    N : int
        The number of locations.
    """

    mu = np.zeros(2*N)
    mu[:N] = theta.mu_0
    mu[N:] = theta.mu_1
    return mu


def compute_Sigma(
        theta: ThetaClass,
        locations: np.ndarray
) -> np.ndarray:
    
    r"""Compute the prior covariance matrix for latent parameters beta (Matern 3/2 kernel)

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    locations : np.ndarray
        The locations where the covariance matrix is computed.
    """

    dist_matrix = np.abs(locations[:, None] - locations[None, :])
    dist_matrix = dist_matrix * np.sqrt(3) / theta.length_scale
    r_matrix = (1.0 + dist_matrix) * np.exp(-dist_matrix)
    R0 = theta.sigma_0**2 * r_matrix
    R1 = theta.sigma_1**2 * r_matrix
    Rco = theta.rho * theta.sigma_0 * theta.sigma_1 * r_matrix
    Sigma = np.block([
        [R0, Rco],
        [Rco, R1]
    ])
    return Sigma


def compute_Sigma_post_inv(
        theta: ThetaClass,
        x_pow: np.ndarray,
        Sigma_inv: np.ndarray
) -> np.ndarray:

    r"""Compute the inverse posterior covariance matrix of the lateent parameters beta.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    x_pow : np.ndarray, shape=(T, N)
        The wind speed raised at power theta.exponent at each location and time point.
    Sigma_inv : np.ndarray, shape=(2*N, 2*N)
        The inverse covariance matrix of the model (precision matrix).
    """

    T, N = x_pow.shape
    x_pow_sq = np.square(x_pow)
    sum_x_pow = np.sum(x_pow, axis=0)
    sum_x_pow_sq = np.sum(x_pow_sq, axis=0)
    inv_tau_sq = 1.0 / np.square(theta.tau)
    A = np.zeros((N, N))
    np.fill_diagonal(A, inv_tau_sq * T)
    B = np.zeros((N, N))
    np.fill_diagonal(B, inv_tau_sq * sum_x_pow)
    C = np.zeros((N, N))
    np.fill_diagonal(C, inv_tau_sq * sum_x_pow_sq)
    FF = np.block([
        [A, B],
        [B, C]
    ])
    Sigma_post_inv = Sigma_inv + FF
    return Sigma_post_inv


def compute_mu_post(
        theta: ThetaClass,
        x_pow: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        Sigma_inv: np.ndarray,
        Sigma_post_inv: np.ndarray
) -> np.ndarray:
    
    r"""Compute the posterior mean of the latent parameters beta.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    x_pow : np.ndarray, shape=(T, N)
        The wind speed raised at power theta.exponent at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    mu : np.ndarray, shape=(2*N,)
        The prior mean of the latent parameters.
    Sigma_inv : np.ndarray, shape=(2*N, 2*N)
        The inverse prior covariance matrix of the latent parameters.
    Sigma_post_inv : np.ndarray, shape=(2*N, 2*N)
        The inverse posterior covariance matrix of the latent parameters.
    """

    T, N = x_pow.shape
    inv_tau_sq = 1.0 / np.square(theta.tau)
    Fy = np.zeros(2*N)
    Fy[:N] = np.sum(y, axis=0)
    Fy[N:] = np.sum(x_pow * y, axis=0)
    mu_post = sp.linalg.solve(a=Sigma_post_inv, b=(Sigma_inv @ mu + inv_tau_sq * Fy), assume_a='pos')
    return mu_post


def nll(
        theta: ThetaClass,
        locations: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
) -> float:

    r"""Compute the negative log-likelihood of parameters theta up to an additive constant w.r.t. theta.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    locations : np.ndarray
        The locations where the covariance matrix is computed.
    x : np.ndarray, shape=(T, N)
        The wind speed at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    """

    T, N = x.shape
    x_pow = x**theta.exponent
    mu = compute_mu(
        theta=theta,
        N=N
    )
    Sigma = compute_Sigma(
        theta=theta,
        locations=locations
    )
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_post_inv = compute_Sigma_post_inv(
        theta=theta,
        x_pow=x_pow,
        Sigma_inv=Sigma_inv
    )
    mu_post = compute_mu_post(
        theta=theta,
        x_pow=x_pow,
        y=y,
        mu=mu,
        Sigma_inv=Sigma_inv,
        Sigma_post_inv=Sigma_post_inv
    )
    inv_tau_sq = 1.0 / np.square(theta.tau)
    mu_diff = mu_post - mu
    model_error = y - mu_post[None, :N] - x_pow * mu_post[None, N:]

    sign_det, log_abs_det = np.linalg.slogdet(Sigma_post_inv)
    assert sign_det == 1.0
    nll = log_abs_det

    sign_det, log_abs_det = np.linalg.slogdet(Sigma)
    assert sign_det == 1.0
    nll += log_abs_det

    nll += np.log(theta.tau) * 2 * N * T
    nll += mu_diff @ Sigma_inv @ mu_diff
    nll += np.sum(np.sum(np.square(model_error))) * inv_tau_sq
    
    return nll


def nlp(
        theta: ThetaClass,
        locations: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        prior: PriorClass
) -> float:
    
    r"""Compute the negative log-posterior of parameters theta up to an additive constant w.r.t. theta.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    locations : np.ndarray
        The locations where the covariance matrix is computed.
    x : np.ndarray, shape=(T, N)
        The wind speed at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    prior : PriorClass
        The prior distribution of the parameters of the model.
    """

    return nll(theta, locations, x, y) - prior.logpdf(theta)


def get_prediction_function(
        theta: ThetaClass,
        locations: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
):

    T, N = x.shape
    x_pow = x**theta.exponent
    mu = compute_mu(
        theta=theta,
        N=N
    )
    Sigma = compute_Sigma(
        theta=theta,
        locations=locations
    )
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_post_inv = compute_Sigma_post_inv(
        theta=theta,
        x_pow=x_pow,
        Sigma_inv=Sigma_inv
    )
    mu_post = compute_mu_post(
        theta=theta,
        x_pow=x_pow,
        y=y,
        mu=mu,
        Sigma_inv=Sigma_inv,
        Sigma_post_inv=Sigma_post_inv
    )

    # finding channel value from channel index
    channel_idx = np.ones((np.max(locations) + 1,), dtype=int) * len(locations)
    channel_idx[locations] = np.arange(len(locations))
    channel_idx = channel_idx.astype(int)

    # computing the prediction
    def compute_pred(x: float, channel: int):
        idx = channel_idx[channel]
        prediction = mu_post[idx] + x**theta.exponent * mu_post[N + idx]
        return prediction
    
    return compute_pred


def generate_data(
        theta: ThetaClass,
        locations: np.ndarray,
        T: int,
        wind: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        noise: Optional[np.ndarray] = None,
        seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    r"""Generate synthetic data from the model.

    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    locations : np.ndarray, shape=(N,)
        The locations of DAS channels.
    T : int
        The number of time points.
    wind : np.ndarray, shape=(T, N), optional
        The wind speed at each location and time point.
    beta : np.ndarray, shape=(2*N,), optional
        The latent parameters of the model.
    noise : np.ndarray, shape=(T, N), optional
        The noise term.

    Returns
    -------
    x : np.ndarray, shape=(T, N)
        The wind speed at each location and time point.
    y : np.ndarray, shape=(T, N)
        The DAS signal at each location and time point.
    beta : np.ndarray, shape=(2*N,)
        The latent parameters of the model.
    """

    rand_gen = np.random.default_rng(seed)
    N, = locations.shape

    if wind is None:
        wind = rand_gen.uniform(low=0.0, high=6.0, size=(T, N))
    x = wind

    if beta is None:
        mu = compute_mu(
            theta=theta,
            N=N
        )
        Sigma = compute_Sigma(
            theta=theta,
            locations=locations
        )
        beta = rand_gen.multivariate_normal(mean=mu, cov=Sigma)
    
    if noise is None:
        noise = rand_gen.normal(loc=0.0, scale=theta.tau, size=(T, N))
    
    x_pow = np.power(x, theta.exponent)
    y = beta[None, :N] + x_pow * beta[None, N:] + noise

    return x, y, beta
