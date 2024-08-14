import numpy as np
import scipy.stats as sps
from typing import Optional


r"""Implementation of a power-law model in numpy.

Extended summary
----------------
The model used here is: 

.. math:: y_{s,t} = \beta_{0} + \beta_{1} x_{s,t}^{\alpha} + \epsilon_{s,t}

where :math:`y_{s,t}` (`y` in the code) is the DAS signal at location :math:`s` and time point :math:`t`,
:math:`x_{s,t}` (`x` in the code) is the wind speed at location :math:`s` and time point :math:`t`,
:math:`\alpha` (`theta.exponent` in the code) is the exponent in the power-law relationship between wind speed and DAS signal,
:math:`\beta{1}` (`theta.slope` in the code) is the slope of the power-law relationship,
:math:`\beta{0}` (`theta.intercept` in the code) is the intercept of the power-law relationship, and
:math:`\epsilon_{s,t}` is a Gaussian white noise prior with variance `\theta.tau^2` and zero mean.

Parameters are stored in a `ThetaClass` object, and the model is implemented as a set of functions.
This module enables the computation of the posterior negative log-likelihood of the model given data 
up to an additive constant w.r.t. to `theta`.
It is intended to be used in a Bayesian optimization setting, where the negative log-likelihood is minimized w.r.t. `theta`.

We use the following conventions.
    N: int
        the number of data points
    x : np.ndarray, shape=(N,)
        the wind speed at each data point.
    x_pow : np.ndarray, shape=(N,)
        the wind speed raised at power theta.exponent at each data point.
    y : np.ndarray, shape=(N,)
        the DAS signal at each location and time point.
    theta : ThetaClass
        the parameters of the model. See `ThetaClass` for details.
    <var>_sq
        the square of <var>, in the sense of element-wise squaring.
"""


class ThetaClass():

    r"""Data class for the parameters of the model.
    
    Parameters
    ----------
    exponent : float
        The exponent in the power-law relationship between wind speed and DAS signal.
    slope : float
        The slope of the power-law relationship.
    intercept : float
        The intercept of the power-law relationship.
    tau : float
        The standard deviation of the Gaussian white noise prior.
    """

    def __init__(
            self,
            exponent: Optional[float] = None,
            slope: Optional[float] = None,
            intercept: Optional[float] = None,
            tau: Optional[float] = None,
    )-> None:    
        self.exponent = exponent
        self.slope = slope
        self.intercept = intercept
        self.tau = tau
    
    def __repr__(
            self
    ) -> str:
        
        return f"powerlaw_numpy ThetaClass object at {hex(id(self))}:\n" \
            + f"    exponent={self.exponent}\n" \
            + f"    slope={self.slope}\n" \
            + f"    intercept={self.intercept}\n" \
            + f"    tau={self.tau}"
    
    def from_array(
            self, 
            theta_array: np.ndarray
    ) -> None:
        
        self.exponent = theta_array[0]
        self.slope = theta_array[1]
        self.intercept = theta_array[2]
        self.tau = theta_array[3]

    def to_array(
            self
    ) -> np.ndarray:
        
        return np.array([self.exponent, self.slope, self.intercept, self.tau])
    
    def to_print_array(
            self
    ) -> np.ndarray:
        
        return np.array([self.exponent, self.intercept, self.slope, self.tau])
    
    def get_denormalized(
            self,
            x_mean: float,
            y_mean: float,
    ) -> 'ThetaClass':
            
        return ThetaClass(
            exponent= self.exponent,
            slope= y_mean * self.slope / np.power(x_mean, self.exponent),
            intercept= self.intercept * y_mean,
            tau= self.tau * y_mean
        )
    
    @staticmethod
    def get_bounds() -> list[tuple[Optional[float], Optional[float]]]:
        
        return [
            (0., None), # exponent
            (0., None), # slope
            (0., None), # intercept
            (0., None)  # tau
        ]
    
    @staticmethod
    def get_parameter_names() -> list[str]:

        return [
            "exponent",
            "slope",
            "intercept",
            "tau"
        ]


def get_default_theta(
        x: np.ndarray,
        y: np.ndarray
) -> ThetaClass:
        
        r"""Get the default parameters of the model.
    
        Parameters
        ----------
        x : np.ndarray, shape=(N,)
            The wind speed at each data point.
        y : np.ndarray, shape=(N,)
            The DAS signal at each data point.
        
        Returns
        -------
        ThetaClass
            The default parameters of the model.
        """

        #eps = np.finfo(np.float64).eps
        return ThetaClass(
            exponent=1.0,
            slope=np.mean(y)/np.mean(x),
            intercept=0.0,
            tau=np.std(y),
        )



class PriorClass():

    r"""Class for the prior distribution of the parameters of the model.

    Parameters are assumed to be independent, 
    and for each parameter, a prior continuous distribution from scipy stats is specified.
    
    Parameters
    ----------
    exponent : 
        The prior distribution of theta.exponent.
    slope :
        The prior distribution of theta.slope.
    intercept :
        The prior distribution of theta.intercept.
    tau :
        The prior distribution of theta.tau.
    """

    def __init__(
            self,
            exponent: Optional[sps.rv_continuous] = None,
            slope: Optional[sps.rv_continuous] = None,
            intercept: Optional[sps.rv_continuous] = None,
            tau: Optional[sps.rv_continuous] = None,
    ) -> None:
        
        self.exponent = exponent
        self.slope = slope
        self.intercept = intercept
        self.tau = tau
    
    def __repr__(
            self
    ) -> str:
        
        return f"powerlaw_numpy PriorClass object at {hex(id(self))}:\n" \
            + f"    exponent={self.exponent}\n" \
            + f"    slope={self.slope}\n" \
            + f"    intercept={self.intercept}\n" \
            + f"    tau={self.tau}"
    
    @staticmethod
    def _get_logpdf_or_zero(
        dist: Optional[sps.rv_continuous],
        value: float
    ) -> float:
            
        return dist.logpdf(value) if dist is not None else 0.
            
    def logpdf(
            self,
            theta: ThetaClass
    ) -> float:
        
        return np.sum([
            self._get_logpdf_or_zero(self.exponent, theta.exponent),
            self._get_logpdf_or_zero(self.slope, theta.slope),
            self._get_logpdf_or_zero(self.intercept, theta.intercept),
            self._get_logpdf_or_zero(self.tau, theta.tau)
        ])


def get_default_prior(
        x: np.ndarray,
        y: np.ndarray
) -> PriorClass:
    
    r"""Get the default prior distribution of the parameters of the model.

    Parameters
    ----------
    x : np.ndarray, shape=(N,)
        The wind speed at each data point.
    y : np.ndarray, shape=(N,)
        The DAS signal at each data point.
    
    Returns
    -------
    PriorClass
        The default prior distribution of the parameters of the model.
    """

    return PriorClass(
        exponent=sps.gamma(a=2., scale=1.),
        slope=sps.lognorm(s=3., scale=np.mean(y)/np.mean(x)),
        intercept=sps.expon(scale=np.mean(y)/3),
        tau=sps.invgamma(a=2., scale=np.std(y))
    )


def forward(
        theta: ThetaClass,
        x: np.ndarray,
        add_noise: bool = False,
        seed: Optional[int] = None
) -> np.ndarray:
    
    r"""Compute the forward model.
    
    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    x : np.ndarray, shape=(N,)
        The wind speed at each data point.
    add_noise : bool, optional
        Whether to add Gaussian white noise to the model. Default is False.
    
    Returns
    -------
    np.ndarray, shape=(N,)
        The DAS signal at each location and time point.
    """
    
    x_pow = np.power(x, theta.exponent)
    y = theta.intercept + theta.slope * x_pow
    if add_noise:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0., scale=theta.tau, size=x.shape)
        y = y + noise
    return y


def nll(
        theta: ThetaClass,
        x: np.ndarray,
        y: np.ndarray
) -> float:
    
    r"""Compute the negative log-likelihood of the model given data.
    
    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    x : np.ndarray, shape=(N,)
        The wind speed at each data point.
    y : np.ndarray, shape=(N,)
        The DAS signal at each location and time point.
    
    Returns
    -------
    float
        The negative log-likelihood of the model given data.
    """
    
    y_pred = forward(theta, x, add_noise=False)
    tau_sq = np.square(theta.tau)
    return 0.5 * ( np.sum(np.square(y - y_pred)) / tau_sq + y.size * np.log(2 * np.pi * tau_sq))


def nlp(
        theta: ThetaClass,
        x: np.ndarray,
        y: np.ndarray,
        prior: PriorClass,
) -> float:
    
    r"""Compute the negative log-posterior of the model given data.
    
    Parameters
    ----------
    theta : ThetaClass
        The parameters of the model.
    x : np.ndarray, shape=(N,)
        The wind speed at each data point.
    y : np.ndarray, shape=(N,)
        The DAS signal at each location and time point.
    prior : PriorClass
        The prior distribution of the parameters of the model.
    
    Returns
    -------
    float
        The negative log-posterior of the model given data.
    """
    
    return nll(theta, x, y) - prior.logpdf(theta)
