import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union
from .ray import Ray

# A medium can be given as
#   - a number:         constant (non-dispersive) index, e.g. 1.0 for air
#   - a material name:  looked up in the instrument's refractive index table
#   - an (nwave,) array: index already sampled at the instrument wavelengths
Medium = Union[float, str, np.ndarray]

@dataclass
class Instrument:
    """Spectral description of an instrument.

    wavelengths: (nwave,) wavelengths [um] at which the instrument's spectral
                 response is sampled. Every ray traced through this
                 instrument carries one column per wavelength.
    response:    (nwave,) spectral response at those wavelengths, or a
                 function response(wavelength). Defaults to a flat response.
    materials:   {name: index} where index is a number, a function
                 n(wavelength), or a pair (wavelength_table, n_table) that is
                 linearly interpolated onto `wavelengths`.

    After construction the instrument holds the refractive index look-up table
    ``n_table`` of shape (nmaterial, nwave); row i belongs to
    ``material_names[i]``.
    """
    wavelengths: np.ndarray
    response: Union[np.ndarray, Callable, None] = None
    materials: Dict[str, object] = field(default_factory = dict)

    n_table: np.ndarray = field(init = False, repr = False)
    material_names: List[str] = field(init = False)

    def __post_init__(self):
        self.wavelengths = np.atleast_1d(np.asarray(self.wavelengths, float))
        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths should be a 1D array")

        if self.response is None:
            self.response = np.ones(self.nwave)
        elif callable(self.response):
            self.response = np.asarray(self.response(self.wavelengths), float)
        else:
            self.response = np.asarray(self.response, float)
        if self.response.shape != (self.nwave,):
            raise ValueError("response should have the same length as wavelengths")

        self.n_table = np.empty((0, self.nwave))
        self.material_names = []
        for name, index in dict(self.materials).items():
            self.add_material(name, index)

    @property
    def nwave(self) -> int:
        return len(self.wavelengths)

    @property
    def weights(self) -> np.ndarray:
        """Spectral response normalized to sum to one, (nwave,)."""
        return self.response / np.sum(self.response)

    def add_material(self, name: str, index) -> None:
        """Add (or replace) a row of the refractive index look-up table."""
        if callable(index):
            row = index(self.wavelengths)
        elif isinstance(index, tuple) and len(index) == 2:
            wave_tab, n_tab = (np.asarray(a, float) for a in index)
            if np.any(self.wavelengths < wave_tab.min()) or \
               np.any(self.wavelengths > wave_tab.max()):
                raise ValueError(f"table for '{name}' does not cover the instrument wavelengths")
            row = np.interp(self.wavelengths, wave_tab, n_tab)
        else:
            row = index
        row = np.broadcast_to(np.asarray(row, float), (self.nwave,))

        if name in self.material_names:
            self.n_table[self.material_names.index(name)] = row
        else:
            self.n_table = np.vstack([self.n_table, row])
            self.material_names.append(name)

    def refractive_index(self, medium: Medium) -> np.ndarray:
        """Refractive index of `medium` at every instrument wavelength, (nwave,)."""
        if isinstance(medium, str):
            if medium not in self.material_names:
                raise KeyError(f"unknown material '{medium}', "
                               f"known materials are {self.material_names}")
            return self.n_table[self.material_names.index(medium)].copy()
        return np.broadcast_to(np.asarray(medium, float), (self.nwave,)).copy()

    def make_ray(self, pos, dir, medium: Medium = 1.) -> Ray:
        """Create a ray with one column per instrument wavelength."""
        return self.expand(Ray(pos = pos, dir = dir), medium)

    def expand(self, ray: Ray, medium: Medium | None = None) -> Ray:
        """Broadcast a (monochromatic) ray onto the instrument wavelengths.

        If `medium` is None the ray's own n_in is kept.
        """
        if ray.nwave not in (1, self.nwave):
            raise ValueError(f"ray has {ray.nwave} wavelengths, "
                             f"instrument has {self.nwave}")
        shape = (ray.ndim, self.nwave)
        n_in = ray.n_in if medium is None else self.refractive_index(medium)
        return Ray(pos = np.broadcast_to(ray.pos, shape),
                   dir = np.broadcast_to(ray.dir, shape),
                   n_in = n_in)

    def weighted_mean(self, x: np.ndarray) -> np.ndarray:
        """Average x over its trailing (wavelength) axis, weighted by the
        instrument response. Lost wavelengths (NaN) are ignored."""
        x = np.asarray(x, float)
        w = np.where(np.isfinite(x), self.weights, 0.)
        return np.nansum(x * w, axis = -1) / np.sum(w, axis = -1)

