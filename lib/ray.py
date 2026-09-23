import numpy as np
from dataclasses import dataclass

@dataclass
class Ray:
    """A bundle of rays, one per wavelength, that share a starting point.

    The trailing dimension of every field is the wavelength dimension:

    - pos:  (ndim, nwave) position of the ray at each wavelength
    - dir:  (ndim, nwave) unit direction at each wavelength
    - n_in: (nwave,)      refractive index of the medium the ray travels in

    A monochromatic ray is simply nwave = 1, so 1-D inputs such as
    ``Ray(pos=[0, 1], dir=[1, 0])`` are promoted to shape (ndim, 1).
    Inputs are broadcast against each other along the wavelength axis.

    A wavelength that has been lost (missed a surface, total internal
    reflection) is marked by NaN in its column of ``pos``.
    """
    pos: np.ndarray
    dir: np.ndarray
    n_in: np.ndarray = 1.

    def __post_init__(self):
        pos = np.asarray(self.pos, float)
        dir = np.asarray(self.dir, float)
        n_in = np.atleast_1d(np.asarray(self.n_in, float))

        if pos.ndim == 1:
            pos = pos[:, None]
        if dir.ndim == 1:
            dir = dir[:, None]
        if pos.ndim != 2 or dir.ndim != 2 or n_in.ndim != 1:
            raise ValueError("pos and dir should be (ndim, nwave), n_in should be (nwave,)")
        if pos.shape[0] <= 1:
            raise ValueError("position vector should be at least 2D")
        if dir.shape[0] <= 1:
            raise ValueError("direction vector should be at least 2D")
        if pos.shape[0] != dir.shape[0]:
            raise ValueError("position and direction must have the same spatial dimension")

        nwave = np.broadcast_shapes((pos.shape[1],), (dir.shape[1],), n_in.shape)[0]
        ndim = pos.shape[0]

        self.pos = np.broadcast_to(pos, (ndim, nwave)).copy()
        dir = np.broadcast_to(dir, (ndim, nwave))
        self.dir = dir / np.linalg.norm(dir, axis=0)
        self.n_in = np.broadcast_to(n_in, (nwave,)).copy()

    @property
    def ndim(self) -> int:
        return self.pos.shape[0]

    @property
    def nwave(self) -> int:
        return self.pos.shape[1]

    @property
    def alive(self) -> np.ndarray:
        """Boolean mask (nwave,) of wavelengths that are still being traced."""
        return np.all(np.isfinite(self.pos), axis=0)

    def __eq__(self, other):
        if not isinstance(other, Ray):
            return False

        return (
            self.pos.shape == other.pos.shape and
            np.allclose(self.pos, other.pos, equal_nan=True) and
            np.allclose(self.dir, other.dir, equal_nan=True) and
            np.allclose(self.n_in, other.n_in, equal_nan=True)
        )

    def advance(self, step):
        """Move forward by `step`, a scalar or an (nwave,) array of distances.

        A NaN step marks that wavelength as lost.
        """
        step = np.asarray(step, float)
        if np.any(step < 0):
            raise ValueError("Ray must advance in the forward direction")
        new_pos = self.pos + step * self.dir
        return Ray(pos = new_pos, dir = self.dir, n_in = self.n_in)
