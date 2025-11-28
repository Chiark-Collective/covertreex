"""GP kernel classes for residual correlation metric.

These kernels define the covariance structure used for residual correlation
distance computation in Vecchia-style GP neighbor selection.

Example
-------
>>> from covertreex import CoverTree
>>> from covertreex.kernels import Matern52
>>>
>>> kernel = Matern52(lengthscale=1.0, variance=1.0)
>>> tree = CoverTree(coords, kernel=kernel)
>>> neighbors = tree.knn(k=50)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Kernel(ABC):
    """Base class for GP kernels.

    Subclasses must implement `__call__` to compute covariance matrices
    and provide `variance` and `lengthscale` attributes.
    """

    # Subclasses should define these as dataclass fields
    variance: float
    lengthscale: float | np.ndarray

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute covariance matrix K(X1, X2).

        Parameters
        ----------
        X1 : ndarray, shape (n1, d)
            First set of points.
        X2 : ndarray, shape (n2, d)
            Second set of points.

        Returns
        -------
        K : ndarray, shape (n1, n2)
            Covariance matrix.
        """
        ...

    @property
    def kernel_type(self) -> int:
        """Kernel type identifier for Rust backend (0=RBF, 1=Matern52)."""
        return 0


@dataclass(frozen=True)
class RBF(Kernel):
    """Radial Basis Function (squared exponential) kernel.

    K(x, y) = variance * exp(-0.5 * ||x - y||² / lengthscale²)

    Parameters
    ----------
    lengthscale : float, default=1.0
        Lengthscale parameter controlling correlation decay.
    variance : float, default=1.0
        Signal variance (amplitude squared).

    Examples
    --------
    >>> kernel = RBF(lengthscale=2.0, variance=1.0)
    >>> K = kernel(X1, X2)
    """

    lengthscale: float = 1.0
    variance: float = 1.0

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        denom = max(self.lengthscale, 1e-12)
        scaled = -0.5 * sq_dist / (denom * denom)
        return float(self.variance) * np.exp(scaled)

    @property
    def kernel_type(self) -> int:
        return 0


@dataclass(frozen=True)
class Matern52(Kernel):
    """Matérn 5/2 kernel.

    K(x, y) = variance * (1 + a + a²/3) * exp(-a)
    where a = sqrt(5) * ||x - y|| / lengthscale

    Parameters
    ----------
    lengthscale : float, default=1.0
        Lengthscale parameter controlling correlation decay.
    variance : float, default=1.0
        Signal variance (amplitude squared).

    Examples
    --------
    >>> kernel = Matern52(lengthscale=2.0, variance=1.0)
    >>> K = kernel(X1, X2)
    """

    lengthscale: float = 1.0
    variance: float = 1.0

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        denom = max(self.lengthscale, 1e-12)
        r = np.sqrt(sq_dist) / denom
        sqrt5 = np.sqrt(5.0)
        a = sqrt5 * r
        poly = 1.0 + a + (a * a) / 3.0
        return float(self.variance) * poly * np.exp(-a)

    @property
    def kernel_type(self) -> int:
        return 1


__all__ = ["Kernel", "RBF", "Matern52"]
