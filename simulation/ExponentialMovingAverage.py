class ExponentialMovingAverage:
    def __init__(self, alpha : float):
        self.alpha = alpha
        self.yLast = 0.0

    def ProcessSample(self, x : float) -> float:
        self.yLast = self.alpha * x + (1 - self.alpha) * self.yLast
        return self.yLast