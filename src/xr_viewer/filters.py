# Desktop2Stereo OpenXR viewer: smoothing filters.

import math

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter for hand-jitter reduction.

    Algorithm (from "One Euro Filter" by Gery Casiez et al.):
    dx  = (x - x_prev) / dt                  raw derivative
    dx^ = low_pass(dx, f_Cd, dt)             smoothed derivative
    f_C = min_cutoff + beta * |dx^|          adaptive cutoff (Hz)
    x^  = low_pass(x, f_C, dt)               filtered output

    The low-pass is a 1st-order RC filter:
    alpha   = 1 / (1 + tau / dt)                   smoothing factor
    tau   = 1 / (2pi * f_C)                     time constant
    y   = alpha*x + (1-alpha)*y_prev

    Tuning guide:
    min_cutoff (Hz):      1.0-.0 for hand tracking. Lower = smoother, more lag.
                            Start at 1.2.
    beta:                 0.007-.05. Speed sensitivity -higher responds faster
                            to quick moves but transmits more jitter. Start at 0.01.
    derivative_cutoff (Hz): 1.0 is typical. Lower = smoother derivative estimate.
    """
    __slots__ = ('min_cutoff', 'beta', 'derivative_cutoff', '_x_prev', '_dx_prev')

    def __init__(self, min_cutoff=1.2, beta=0.01, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._x_prev = None    # previous filtered value
        self._dx_prev = None   # previous smoothed derivative

    def reset(self):
        self._x_prev = None
        self._dx_prev = None

    def _alpha(self, cutoff, dt):
        if dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt):
        if dt <= 0.0 or self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return float(x)

        # Derivative of the raw signal
        dx = (float(x) - self._x_prev) / dt

        # Smooth the derivative with fixed cutoff
        alpha_d = self._alpha(self.derivative_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff: rises with speed ->less lag during fast motion
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Low-pass with adaptive cutoff
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * float(x) + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class EMAPositionFilter:
    """Simple exponential moving average -fallback for debugging.

    y = alpha*x + (1-alpha)*y_prev
    """
    __slots__ = ('alpha', '_prev')

    def __init__(self, alpha=0.15):
        self.alpha = float(alpha)
        self._prev = None

    def reset(self):
        self._prev = None

    def filter(self, x):
        if self._prev is None:
            self._prev = float(x)
            return float(x)
        x_hat = self.alpha * float(x) + (1.0 - self.alpha) * self._prev
        self._prev = x_hat
        return x_hat


class OneEuroFilter3D:
    """Three independent One Euro Filters for 3D position (X, Y, Z)."""
    __slots__ = ('_fx', '_fy', '_fz')

    def __init__(self, min_cutoff=1.2, beta=0.01, derivative_cutoff=1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._fz = OneEuroFilter(min_cutoff, beta, derivative_cutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()

    def filter(self, pos, dt):
        x = self._fx.filter(float(pos[0]), dt)
        y = self._fy.filter(float(pos[1]), dt)
        z = self._fz.filter(float(pos[2]), dt)
        return np.array([x, y, z], dtype='f8')