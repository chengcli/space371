import numpy as np
from dataclasses import dataclass

@dataclass
class Ray:
    pos: np.ndarray
    dir: np.ndarray
    n_in: float = 1.

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        if len(self.pos) <= 1:
            raise ValueError("position vector should be at least 2D")
        dir = np.asarray(self.dir, float)
        self.dir = dir / np.linalg.norm(dir)
        if len(self.dir) <= 1:
            raise ValueError("direction vector should be at least 2D")

    def __eq__(self, other):
        if not isinstance(other, Ray):
            return false
        
        return (
            np.allclose(self.pos, other.pos) and 
            np.allclose(self.dir, other.dir) and 
            np.isclose(self.n_in, other.n_in)
        )

    def advance(self, step: float):
        if step < 0:
            raise ValueError("Ray must advance in the forward direction")
        new_pos = self.pos + step * self.dir
        return Ray(pos = new_pos, dir = self.dir, n_in = self.n_in)
