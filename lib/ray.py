import numpy as np
from dataclasses import dataclass

@dataclass
class Ray:
    pos: np.ndarray
    dir: np.ndarray
    n_in: np.ndarray = 1.

    def __post_init__(self):
        # pos and dir are (ndim, nwave), n_in is (nwave,)
        self.pos = np.asarray(self.pos, float).reshape(len(self.pos), -1)
        if len(self.pos) <= 1:
            raise ValueError("position vector should be at least 2D")
        dir = np.asarray(self.dir, float).reshape(len(self.dir), -1)
        self.dir = dir / np.linalg.norm(dir, axis = 0)
        if len(self.dir) <= 1:
            raise ValueError("direction vector should be at least 2D")

    def __eq__(self, other):
        if not isinstance(other, Ray):
            return False
        
        return (
            np.allclose(self.pos, other.pos) and 
            np.allclose(self.dir, other.dir) and 
            np.allclose(self.n_in, other.n_in)
        )

    def advance(self, step: np.ndarray):
        if np.any(step < 0):
            raise ValueError("Ray must advance in the forward direction")
        new_pos = self.pos + step * self.dir
        return Ray(pos = new_pos, dir = self.dir, n_in = self.n_in)
