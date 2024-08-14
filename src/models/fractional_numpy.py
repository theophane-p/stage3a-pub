import numpy as np
import scipy.stats as sps
from typing import Optional

"""Implementation of a generalised linear model with a rational basis function with numpy.

Extended summary
----------------
The model used here is: 

.. math:: y_{t,n} = \beta_{0} + \beta_{1} \frac{(x_{s,t} / \beta{2})^{\alpha}}{1 + (x_{s,t} / \beta{2})^{\alpha}} + \epsilon_{s,t}

where :math:`y_{s,t}` (`y` in the code) is the DAS signal at location :math:`s` and time point :math:`t`,
:math:`x_{s,t}` (`x` in the code) is the wind speed at location :math:`s` and time point :math:`t`,
:math:`\alpha` (`theta.exponent` in the code) is the exponent in the power-law relationship between wind speed and DAS signal,
:math:`\beta{0}` (`theta.add_const` in the code) is the additive constant,
:math:`\beta{1}` (`theta.mul_const` in the code) is the multiplication constant for the rational function,
:math:`\beta{2}` (`theta.x_scale` in the code) is the inverse scaling factor for the wind speed, and
:math:`\epsilon_{t,n}` is a Gaussian white noise prior with variance `\theta.tau^2` and zero mean.

Parameters are stored in a `ThetaClass` object, and the model is implemented as a set of functions.
This module enables the computation of the posterior negative log-likelihood of the model given data 
up to an additive constant w.r.t. to `theta`.
It is intended to be used in a Bayesian optimization setting, where the negative log-likelihood is minimized w.r.t. `theta`.

We use the following conventions.
    N: int
        the number of data points
    x : np.ndarray, shape=(N,)
        the wind speed at each data point.
    x_bf : np.ndarray, shape=(N,)
        the basis function applied to the wind speed at each data point.
    y : np.ndarray, shape=(N,)
        the DAS signal at each location and time point.
    theta : ThetaClass
        the parameters of the model. See `ThetaClass` for details.
    <var>_sq
        the square of <var>, in the sense of element-wise squaring.
"""


class ThetaClass:

    r"""Data class for the parameters of the model.
    
    Parameters
    ----------
    exponent : float
        The exponent in the power-law relationship between wind speed and DAS signal.
    add_const : float
        The additive constant
    mul_const : float
        The multiplication constant for the rational function
    x_scale : float
        The inverse scaling factor for the wind speed
    tau : float
        The standard deviation of the Gaussian white noise prior.
    """

    def __init__(
            self,
            exponent: Optional[float] = None,
            add_const: Optional[float] = None,
            mul_const: Optional[float] = None,
            x_scale: Optional[float] = None,
            tau: Optional[float] = None,
    ) -> None:
        self.exponent = exponent
        self.add_const = add_const
        self.mul_const = mul_const
        self.x_scale = x_scale
        self.tau = tau

    def __repr__(self) -> str:
        return f"fractional_numpy ThetaClass object at {hex(id(self))}:\n" \
            + f"    exponent={self.exponent}\n" \
            + f"    add_const={self.add_const}\n" \
            + f"    mul_const={self.mul_const}\n" \
            + f"    x_scale={self.x_scale}\n" \
            + f"    tau={self.tau}"

    def from_array(self, theta_array: np.ndarray) -> None:
        self.exponent = theta_array[0]
        self.add_const = theta_array[1]
        self.mul_const = theta_array[2]
        self.x_scale = theta_array[3]
        self.tau = theta_array[4]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.exponent, 
            self.add_const, 
            self.mul_const, 
            self.x_scale, 
            self.tau
        ])
    
    def to_print_array(
            self
    ) -> np.ndarray:
        
        return np.array([
            self.exponent,
            self.x_scale,
            self.add_const,
            self.mul_const,
            self.tau
        ])
    
    def get_denormalized(
            self,
            x_mean: float,
            y_mean: float
    ) -> 'ThetaClass':
        
        return ThetaClass(
            exponent= self.exponent,
            add_const= y_mean * self.add_const,
            mul_const= y_mean * self.mul_const,
            x_scale= x_mean * self.x_scale,
            tau= y_mean * self.tau
        )
    
    @staticmethod
    def get_bounds():
        
        return [
            (0., None), # exponent
            (0., None), # add_const
            (0., None), # mul_const
            (0., None), # x_scale
            (0., None)  # tau
        ]
    
    @staticmethod
    def get_parameter_names() -> list[str]:
        
        return [
            "exponent",
            "add_const",
            "mul_const",
            "x_scale",
            "tau"
        ]



def get_default_theta(
        x: np.ndarray,
        y: np.ndarray
) -> ThetaClass:
        
        r"""Compute the default parameters of the model.
        
        Parameters
        ----------
        x : np.ndarray, shape=(N,)
            The wind speed at each data point.
        y : np.ndarray, shape=(N,)
            The DAS signal at each location and time point.
        
        Returns
        -------
        ThetaClass
            The default parameters of the model.
        """

        #eps = np.finfo(np.float64).eps
        return ThetaClass(
            exponent=1.0,
            add_const=0.0,
            mul_const=np.mean(y)/np.mean(x),
            x_scale=np.exp(np.mean(np.log(x))),
            tau=np.std(y)
        )


class PriorClass:

    r"""Class for the prior distribution of the parameters of the model.

        Parameters are assumed to be independent, 
        and for each parameter, a prior continuous distribution from scipy stats is specified.

    Parameters
    ----------
    exponent :
        The prior distribution of the theta.exponent parameter.
    add_const :
        The prior distribution of the theta.add_const parameter.
    mul_const :
        The prior distribution of the theta.mul_const parameter.
    x_scale :
        The prior distribution of the theta.x_scale parameter.
    tau :
        The prior distribution of the theta.tau parameter.
    """

    def __init__(
            self,
            exponent: Optional[sps.rv_continuous] = None,
            add_const: Optional[sps.rv_continuous] = None,
            mul_const: Optional[sps.rv_continuous] = None,
            x_scale: Optional[sps.rv_continuous] = None,
            tau: Optional[sps.rv_continuous] = None,
    ):
        self.exponent = exponent
        self.add_const = add_const
        self.mul_const = mul_const
        self.x_scale = x_scale
        self.tau = tau
    
    def __repr__(self) -> str:
        return f"fractional_numpy PriorClass object at {hex(id(self))}:\n" \
            + f"    exponent={self.exponent}\n" \
            + f"    add_const={self.add_const}\n" \
            + f"    mul_const={self.mul_const}\n" \
            + f"    x_scale={self.x_scale}\n" \
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
            self._get_logpdf_or_zero(self.add_const, theta.add_const),
            self._get_logpdf_or_zero(self.mul_const, theta.mul_const),
            self._get_logpdf_or_zero(self.x_scale, theta.x_scale),
            self._get_logpdf_or_zero(self.tau, theta.tau)
        ])


def get_default_prior(
        x: np.ndarray,
        y: np.ndarray
) -> PriorClass:
    
    r"""Compute the default prior distribution of the parameters of the model.
    
    Parameters
    ----------
    x : np.ndarray, shape=(N,)
        The wind speed at each data point.
    y : np.ndarray, shape=(N,)
        The DAS signal at each location and time point.
    
    Returns
    -------
    PriorClass
        The default prior distribution of the parameters of the model.
    """
    
    logx = np.log(x[x > 0.])
    return PriorClass(
        exponent=sps.gamma(a=2., scale=2.),
        add_const=sps.expon(scale=np.mean(y)/3),
        mul_const=sps.lognorm(s=3., scale=np.mean(y)/np.mean(x)),
        tau=sps.invgamma(a=2., scale=np.std(y)),
        x_scale=sps.lognorm(s=2*np.std(logx), scale=np.exp(np.mean(logx)))
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
        Whether to add noise to the prediction.
    
    Returns
    -------
    np.ndarray, shape=(N,)
        The predicted DAS signal at each location and time point.
    """

    x_scaled = x * (1. / theta.x_scale)
    x_pow = np.power(x_scaled, theta.exponent)
    y = theta.add_const + theta.mul_const * x_pow / (1. + x_pow)
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
    return 0.5 * (np.sum(np.square(y - y_pred)) / tau_sq + y.size * np.log(2 * np.pi * tau_sq))


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

