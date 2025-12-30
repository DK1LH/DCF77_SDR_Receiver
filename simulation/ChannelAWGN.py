import numpy as np

def ChannelAWGN(x: np.ndarray, snr_db: float, rng=None, axis=None):
    """
    Add AWGN to signal x to obtain a specified SNR (in dB).

    SNR is defined as:
        SNR = P_signal / P_noise
    where P_signal = mean(|x|^2) and P_noise = mean(|n|^2).

    Parameters
    ----------
    x : np.ndarray
        Input signal (real or complex).
    snr_db : float
        Target SNR in dB.
    rng : None | int | np.random.Generator
        Random seed or Generator for reproducibility.
    axis : None | int | tuple[int]
        If None: use total average power over all samples.
        If set: compute power along this axis (keeps dims) so each "channel"
        (e.g. rows or columns) gets its own noise scaling.

    Returns
    -------
    y : np.ndarray
        Noisy signal, same shape/dtype (complex stays complex).
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