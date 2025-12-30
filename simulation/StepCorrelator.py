import numpy as np
from collections import deque
from typing import Union

class StepCorrelator:
    """
    Step correlator with kernel [-1..-1, +1..+1] (lengthStep each side).
    Supports:
      - Block mode:   ProcessSamples(x)  -> y (same length, centered output)
      - Streaming:    PushSample(v)      -> y_value (aligned to newest sample), NaN until ready
    """
    def __init__(self, lengthStep: int, normalize: bool = True):
        if lengthStep <= 0:
            raise ValueError("lengthStep must be > 0")
        self.lengthStep = int(lengthStep)
        self.normalize = bool(normalize)

        # Streaming state (two windows of size lengthStep)
        self._left: deque[float] = deque()
        self._right: deque[float] = deque()
        self._left_sum: float = 0.0
        self._right_sum: float = 0.0

    # ------------------------
    # Streaming / sliding mode
    # ------------------------
    def Reset(self) -> None:
        self._left.clear()
        self._right.clear()
        self._left_sum = 0.0
        self._right_sum = 0.0

    def PushSample(self, v: Union[float, int]) -> float:
        """
        Push one new sample. Returns:
          - np.nan until 2*lengthStep samples have been accumulated
          - then the step correlation value for the boundary between left/right windows
            (aligned to the newest sample at the end of the right window)
        """
        v = float(v)
        M = self.lengthStep

        # Fill left window first
        if len(self._left) < M:
            self._left.append(v)
            self._left_sum += v
            return np.nan

        # Fill right window
        if len(self._right) < M:
            self._right.append(v)
            self._right_sum += v
            if len(self._right) < M:
                return np.nan
            return self._current_value()

        # Steady-state slide by 1
        old_left = self._left.popleft()
        self._left_sum -= old_left

        moved = self._right.popleft()   # moves from right -> left
        self._right_sum -= moved
        self._left.append(moved)
        self._left_sum += moved

        self._right.append(v)          # new sample enters right
        self._right_sum += v

        return self._current_value()

    def PushSamples(self, x: np.ndarray) -> np.ndarray:
        """
        Convenience: push an array in streaming mode and get one output per input.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty(x.size, dtype=float)
        for i, v in enumerate(x):
            out[i] = self.PushSample(v)
        return out

    def _current_value(self) -> float:
        y = self._right_sum - self._left_sum
        return y / self.lengthStep if self.normalize else y

    # ------------------------
    # Block mode (your original)
    # ------------------------
    def ProcessSamples(self, x: np.ndarray) -> np.ndarray:
        """
        Block processing, centered output:
          y[k] = mean(x[k:k+M]) - mean(x[k-M:k]) for k in [M, N-M)
        """
        x = np.asarray(x, dtype=float)
        N = x.size
        y = np.zeros(N, dtype=float)
        M = self.lengthStep

        if N < 2 * M + 1:
            return y

        p = np.zeros(N + 1, dtype=float)
        p[1:] = np.cumsum(x)

        def sum_range(a: int, b: int) -> float:  # [a, b)
            return p[b] - p[a]

        scale = float(M) if self.normalize else 1.0

        for k in range(M, N - M):
            left = sum_range(k - M, k)
            right = sum_range(k, k + M)
            y[k] = (right - left) / scale

        return y
