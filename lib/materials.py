import numpy as np

def sellmeier(B, C):
    """Return n(wavelength) from the Sellmeier equation, wavelength in um:

        n^2 = 1 + sum_i B_i lambda^2 / (lambda^2 - C_i),   C_i in um^2
    """
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    def index(wavelength):
        lam2 = np.asarray(wavelength, float)[..., None] ** 2
        return np.sqrt(1. + np.sum(B * lam2 / (lam2 - C), axis = -1))

    return index

# Schott N-BK7 (n_d = 1.5168 at 0.5876 um)
BK7 = sellmeier(B = [1.03961212, 0.231792344, 1.01046945],
                C = [6.00069867e-3, 2.00179144e-2, 103.560653])

# Schott N-SF11, a dense flint (n_d = 1.7847)
SF11 = sellmeier(B = [1.73759695, 0.313747346, 1.89878101],
                 C = [0.013188707, 0.0623068142, 155.23629])
