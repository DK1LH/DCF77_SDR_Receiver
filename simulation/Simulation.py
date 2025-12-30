import numpy as np
import matplotlib.pyplot as plt

from ChannelAWGN import ChannelAWGN
from Goertzel import GoertzelFilter
from StepCorrelator import StepCorrelator

## Parameter Section
SIM_DURATION = 4
FS = 24e3
TS = (1 / FS)
AWGN_CHANNEL_SNR = -3
GOERTZEL_ANALYZATION_DURATION = 0.01
ANALYZATION_BATCH_LENGTH = int(np.round(GOERTZEL_ANALYZATION_DURATION / TS))
STEP_CORRELATION_STEP_LENGTH = 4


## DCF77 Signal Creation
t = np.arange(0, SIM_DURATION, TS)
dcf77_carrier = np.cos(2 * np.pi * 77.5e3 * t)

# Preparing the symbols representing a '1' and a '0'
sym_one = np.concatenate([np.zeros(int(np.round(0.2 * FS))), np.ones(int(np.round(0.8 * FS)))])
sym_zero = np.concatenate([np.zeros(int(np.round(0.1 * FS))), np.ones(int(np.round(0.9 * FS)))]) 

# Generating the timecode signal by alternating the symbols
timecode_signal = []
for i in range(0, int(np.ceil(SIM_DURATION))):
    if np.mod(i, 2) == 0:
        timecode_signal = np.concatenate([timecode_signal, sym_one])
    else:
        timecode_signal = np.concatenate([timecode_signal, sym_zero])
timecode_signal = timecode_signal[:t.size]

# Modulating the carrier by multiplying with the scaled timecode signal
dcf77_signal = dcf77_carrier * (0.85 * timecode_signal + 0.15)

# Passing the signal through an AWGN-Channel
dcf77_signal_awgn = ChannelAWGN(dcf77_signal, AWGN_CHANNEL_SNR)

# Splitting the signal in batches
dcf77_signal_last_sample_idx = (len(dcf77_signal_awgn) // ANALYZATION_BATCH_LENGTH) * ANALYZATION_BATCH_LENGTH
dcf77_batches = list(dcf77_signal_awgn[:dcf77_signal_last_sample_idx].reshape(-1, ANALYZATION_BATCH_LENGTH))


## Processing Section
goertzelFilter = GoertzelFilter(5.5e3, FS, ANALYZATION_BATCH_LENGTH)
stepCorrelator = StepCorrelator(STEP_CORRELATION_STEP_LENGTH, True)

goertzelResults = np.empty(len(dcf77_batches))
stepCorrelatorResults = np.empty(len(dcf77_batches))
for i in range(len(dcf77_batches)):
    goertzelResults[i] = goertzelFilter.ProcessSamples(dcf77_batches[i])
    stepCorrelatorResults[i] = stepCorrelator.PushSample(goertzelResults[i])


## Plotting Section
fig, axs = plt.subplots(5)
plt.setp(axs, xlim=[0, SIM_DURATION])
axs[0].plot(t, timecode_signal)
axs[1].plot(t, dcf77_signal)
axs[2].plot(t, dcf77_signal_awgn)

t_batch = np.arange(0, SIM_DURATION, GOERTZEL_ANALYZATION_DURATION)
axs[3].stem(t_batch, goertzelResults)
axs[4].stem(t_batch, stepCorrelatorResults)
plt.show()