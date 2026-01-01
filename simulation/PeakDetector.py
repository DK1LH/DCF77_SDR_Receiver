import numpy as np

from ExponentialMovingAverage import ExponentialMovingAverage

class PeakDetector:
    """
    Docstring for PeakDetector

    Very basic online peak detector.
    Uses a scaling factor and an exponential moving average to build a threshold and three samples to detect a peak.
    """

    def __init__(self, emaAlpha : float, thresholdScaling : float):
        """
        Docstring for __init__
        
        :param self: Description
        :param emaAlpha: Feedback factor of the exponential moving average IIR-Filter.
        :type emaAlpha: float
        :param thresholdScaling: Scaling factor of the threshold based on the exponential moving average.
        :type thresholdScaling: float
        """
        
        self.ema = ExponentialMovingAverage(emaAlpha)
        self.thresholdScaling = thresholdScaling
        self.lastSamples = []

    def ProcessSample(self, x : float) -> float:
        """
        Docstring for ProcessSample
        
        Function for processing the newest sample and getting the result whether a positive or negative peak was detected.

        :param self: Description
        :param x: Input sample.
        :type x: float
        :return: 0 if no peak was detected. 1 if a positive peak was detected. -1 if a negative peak was detected.
        :rtype: float
        """
        
        # Feeding the EMA and aborting whether the current sample is below the current threshold
        xAbs = np.abs(x)
        currThreshold = self.ema.ProcessSample(self.thresholdScaling * xAbs)
        if xAbs < currThreshold:
            self.lastSamples.clear()
            return 0
        
        # Keeping only the newest three samples
        if len(self.lastSamples) >= 3:
            self.lastSamples.pop(0)
        self.lastSamples.append(xAbs)

        # Returning a zero if there are not enough samples
        if len(self.lastSamples) < 3:
            return 0

        # Checking whether a peak is present (sample before and after the sample in the middle are smaller) and returning its sign
        if self.lastSamples[0] < self.lastSamples[1] and self.lastSamples[2] < self.lastSamples[1]:
            return np.sign(x)
        
        return 0