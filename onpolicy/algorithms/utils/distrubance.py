import numpy as np

def disturbance_process(n, t, std=0.1, decay_rate=0.1):
    """
    Generate a decaying disturbance vector w_t = std * N(0,1) * exp(-decay_rate * t)

    Args:
        n: Dimension of the disturbance vector
        t: Time step
        std: Standard deviation of the disturbance
        decay_rate: Decay rate of the disturbance
    """
    w = std * np.random.randn(n)      # Random Gaussian
    w *= np.exp(-decay_rate * t)  # Exponential decay
    return w