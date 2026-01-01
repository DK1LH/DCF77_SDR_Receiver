import numpy as np

class GoertzelFilter():
    """
    Docstring for GoertzelFilter

    Simple class for implementing a floating-point Goertzel Filter.
    """

    def __init__(self, analyzationFreq : float, samplingFreq : int, numSamples : int):
        """
        Docstring for __init__

        Initializing the Goertzel Filter. Calculation of the Goertzel coefficient is done in this step.

        :param self: Description
        :param analyzationFreq: Frequency to be analyzed in Hz.
        :type analyzationFreq: float
        :param samplingFreq: Sampling frequency in Hz.
        :type samplingFreq: int
        :param numSamples: Number of samples to be analyzed per batch.
        :type numSamples: int
        """
        
        self.analyzationFreq = analyzationFreq
        self.samplingFreq = samplingFreq
        self.numSamples = numSamples

        # Calculating the Gortzel coefficient
        segmentDuration = numSamples * (1 / samplingFreq)
        k = np.round(analyzationFreq * segmentDuration)
        omega = 2 * np.pi * k / numSamples
        self.goertzelCoeff = 2 * np.cos(omega)

    def ProcessSamples(self, sampleBatch : np.ndarray) -> float:
        """
        Docstring for ProcessSamples
        
        Method to process one batch of samples and to return the absolute amplitude of the frequency to be analyzed.

        :param self: Description
        :param sampleBatch: Batch of input samples.
        :type sampleBatch: np.ndarray
        :return: Absolute amplitude of the frequency to be analyzed.
        :rtype: float
        """
        
        Q0 = 0.0
        Q1 = 0.0
        Q2 = 0.0

        # Processing all samples
        for i in range(self.numSamples):
            Q0 = Q1 * self.goertzelCoeff - Q2 + sampleBatch[i]
            Q2 = Q1
            Q1 = Q0

        # Returning the absolute value of the amplitude
        return np.sqrt((Q1 * Q1) + (Q2 * Q2) - (Q1 * Q2 * self.goertzelCoeff))