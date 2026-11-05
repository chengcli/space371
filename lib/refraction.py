import numpy as np
from .ray import Ray


def calc_reflectivity(ray_in: Ray, norm: np.ndarray, n_after: np.ndarray) -> np.ndarray:
    """Return Fresnel power reflectivities [Rs, Rp] for lossless media.

    s/p fractions use the common plane of incidence of the 2D demos.
    Phase shifts and transport of polarization bases in 3D are not modeled.
    """
    n1, n2 = ray_in.n_in, np.asarray(n_after, float)
    ci = np.clip(np.abs(np.sum(norm * ray_in.dir, axis=0)), 0., 1.)
    k = 1. - (n1 / n2)**2 * (1. - ci**2)
    ct = np.sqrt(np.maximum(k, 0.))
    # At grazing incidence unequal media reflect fully; equal media do not.
    rs = np.ones_like(k)
    rp = np.ones_like(k)
    np.divide(n1 * ci - n2 * ct, n1 * ci + n2 * ct,
              out=rs, where=(n1 * ci + n2 * ct) != 0)
    np.divide(n2 * ci - n1 * ct, n2 * ci + n1 * ct,
              out=rp, where=(n2 * ci + n1 * ct) != 0)
    return np.where(n1 == n2, 0., np.where(k <= 0, 1., np.array([rs**2, rp**2])))


def _split_power(ray, factors):
    power = ray.intensity * ray.polarization * factors
    intensity = np.sum(power, axis=0)
    fractions = np.full_like(power, 0.5)
    np.divide(power, intensity, out=fractions, where=intensity > 0)
    return intensity, fractions


def calc_refraction(ray_in: Ray, norm: np.ndarray, n_after: np.ndarray) -> None | Ray:
    mu = ray_in.n_in / np.asarray(n_after, float)
    c1 = -np.sum(norm * ray_in.dir, axis=0)
    k = 1 - mu * mu * (1 - c1 * c1)
    k = np.where(k < 0, np.nan, k)  # no transmitted direction under TIR
    if np.all(np.isnan(k)):
        return None
    intensity, polarization = _split_power(ray_in, 1 - calc_reflectivity(ray_in, norm, n_after))
    return Ray(pos=ray_in.pos, dir=mu * ray_in.dir + (mu * c1 - np.sqrt(k)) * norm,
               n_in=n_after, intensity=intensity, polarization=polarization)


def calc_reflection(ray_in: Ray, norm: np.ndarray, n_after: np.ndarray) -> Ray:
    c1 = np.sum(norm * ray_in.dir, axis=0)
    intensity, polarization = _split_power(ray_in, calc_reflectivity(ray_in, norm, n_after))
    return Ray(pos=ray_in.pos, dir=ray_in.dir - 2 * c1 * norm,
               n_in=ray_in.n_in, intensity=intensity, polarization=polarization)
