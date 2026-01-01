from collections import deque
from typing import Union

class StepCorrelator:
    """
    Docstring for StepCorrelator

    Online Step correlator for finding falling and rising edges of a signal.
    """

    def __init__(self, lengthStep: int, normalize: bool = True):
        """
        Docstring for __init__
        
        :param lengthStep: Length of the correlation function.
        :type lengthStep: int
        :param normalize: Normalize correlation function for decupling the result from the correlation function's length.
        :type normalize: bool
        """

        if lengthStep <= 0:
            raise ValueError("lengthStep must be > 0")
        self.lengthStep = int(lengthStep)
        self.normalize = bool(normalize)

        # Streaming state (two windows of size lengthStep)
        self._left: deque[float] = deque()
        self._right: deque[float] = deque()
        self._left_sum: float = 0.0
        self._right_sum: float = 0.0


    def Reset(self) -> None:
        self._left.clear()
        self._right.clear()
        self._left_sum = 0.0
        self._right_sum = 0.0

    def ProcessSample(self, v: Union[float, int]) -> float:
        """
        Docstring for ProcessSample

        Function to process a new sample and get the current step correlation output value.

        :param v: New input sample.
        :type v: Union[float, int]
        :return: Step correlation value for the boundary between left/right windows (aligned to the newest sample at the end of the right window).
        :rtype: float
        """

        v = float(v)
        M = self.lengthStep

        # Fill left window first
        if len(self._left) < M:
            self._left.append(v)
            self._left_sum += v
            return 0

        # Fill right window
        if len(self._right) < M:
            self._right.append(v)
            self._right_sum += v
            if len(self._right) < M:
                return 0
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

    def _current_value(self) -> float:
        """
        Docstring for _current_value
        
        :return: Current step correlation value.
        :rtype: float
        """

        y = self._right_sum - self._left_sum
        return y / self.lengthStep if self.normalize else y