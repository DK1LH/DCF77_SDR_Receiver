import numpy as np

from ExponentialMovingAverage import ExponentialMovingAverage

class PeakDetector:
    def __init__(self, emaAlpha : float, thresholdScaling : float):
        self.ema = ExponentialMovingAverage(emaAlpha)
        self.thresholdScaling = thresholdScaling
        self.lastSamples = []

    def ProcessSample(self, x : float) -> float:
        currThreshold = self.ema.ProcessSample(np.abs(self.thresholdScaling * x))
        xAbs = np.abs(x)
        if xAbs < currThreshold:
            self.lastSamples.clear()
            return 0
        
        if len(self.lastSamples) >= 3:
            self.lastSamples.pop(0)
        self.lastSamples.append(xAbs)

        if len(self.lastSamples) < 3:
            return 0

        if self.lastSamples[0] < self.lastSamples[1] and self.lastSamples[2] < self.lastSamples[1]:
            return np.sign(x)
        
        return 0