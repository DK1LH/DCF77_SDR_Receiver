class ExponentialMovingAverage:
    """
    Docstring for ExponentialMovingAverage

    Simple class for implementing an exponential moving average with a single-tap IIR-filter.
    """

    def __init__(self, alpha : float):
        """
        Docstring for __init__
        
        :param self: Description
        :param alpha: Multiplication factor of x[n]. y[n-1] is multiplied by (1 - alpha).
        :type alpha: float
        """

        self.alpha = alpha
        self.yLast = 0.0

    def ProcessSample(self, x : float) -> float:
        """
        Docstring for ProcessSample
        
        Function for processing a single sample and getting the new output value.
        Sample is processed by the formula:
        y[n] = x * alpha + y[n-1] * (1 - alpha)

        :param self: Description
        :param x: Input sample.
        :type x: float
        :return: New output value;
        :rtype: float
        """

        self.yLast = self.alpha * x + (1 - self.alpha) * self.yLast
        return self.yLast