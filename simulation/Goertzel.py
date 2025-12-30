import numpy as np

class GoertzelFilter():
    def __init__(self, analyzationFreq, samplingFreq, numSamples):
        self.analyzationFreq = analyzationFreq
        self.samplingFreq = samplingFreq
        self.numSamples = numSamples

        segmentDuration = numSamples * (1 / samplingFreq)
        k = np.round(analyzationFreq * segmentDuration)
        omega = 2 * np.pi * k / numSamples
        self.goertzelCoeff = 2 * np.cos(omega)

    def ProcessSamples(self, sampleBatch) -> float:
        Q0 = 0.0
        Q1 = 0.0
        Q2 = 0.0

        for i in range(self.numSamples):
            Q0 = Q1 * self.goertzelCoeff - Q2 + sampleBatch[i]
            Q2 = Q1
            Q1 = Q0

        return np.sqrt((Q1 * Q1) + (Q2 * Q2) - (Q1 * Q2 * self.goertzelCoeff))