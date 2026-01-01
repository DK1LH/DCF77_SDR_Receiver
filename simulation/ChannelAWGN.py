import numpy as np

def ChannelAWGN(x: np.ndarray, snr_db: float, rng=None, axis=None) -> np.ndarray:
    """
    Adding AWGN to a signal to obtain a specified SNR

    :param x: Real input signal
    :type x: np.ndarray
    :param snr_db: Desired output SNR in dB
    :type snr_db: float
    :param rng: Random seed or Generator
    :param axis: Leave none to average over all samples
    :return: Array after AWGN-Channel
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """

    x = np.asarray(x)

    # Random generator handling
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be None, an int seed, or a np.random.Generator")

    snr_lin = 10.0 ** (snr_db / 10.0)

    # Signal power: mean(|x|^2)
    sig_power = np.mean(np.abs(x) ** 2, axis=axis, keepdims=(axis is not None))

    # Desired noise power
    noise_power = sig_power / snr_lin
    
    sigma = np.sqrt(noise_power)
    n = sigma * rng.standard_normal(size=x.shape)

    return x + n