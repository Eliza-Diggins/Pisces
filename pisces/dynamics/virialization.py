"""
Virialization methods for initializing particle velocities.
"""
from typing import Literal, Tuple, Union

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import Unit, unyt_array, unyt_quantity

from pisces.dynamics.eddington_sample import generate_velocities
from pisces.dynamics.utils import relative_potential


def compute_eddington_distribution(
    density: Union[unyt_array, np.ndarray],
    potential: Union[unyt_array, np.ndarray],
    npoints: int = 1000,
    boundary_value: Union[unyt_quantity, float] = 0,
    decreasing: bool = False,
    base_units: Union[str, Unit] = "cm**2/s**2",
    base_density_units: Union[str, Unit] = "g/cm**3",
) -> Tuple[unyt_array, unyt_array]:
    r"""
    Compute the Eddington distribution function from a 1D density and potential pair assuming ergodicity and
    isotropy.

    The distribution function :math:`f(\mathcal{E})` is defined as:

    .. math::

        f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2} \fac{d}{d\mathcal{E}} \int_0^\mathcal{E} \frac{1}{\sqrt{\mathcal{E}-\Psi}} \frac{d\rho}{d\psi} d\psi.

    Parameters
    ----------
    density: np.ndarray or unyt_array
        An array of density values from which to determine the distribution function. The array may be in any order; however,
        it must match the ordering of the ``potential`` provided (i.e. as density-potential pairs). If units are not provided,
        then ``base_density_units`` are assumed. By default, the base units are :math:`{\rm g/cm^3}`.
    potential: np.ndarray or unyt_array
        The gravitational potential corresponding to the values in the ``density`` array. If units are not provided,
        then ``base_units`` are assumed. By default, the base units are :math:`{\rm cm^2 s^{-2}}`. The potential array must
        be either monotonically increasing or monotonically decreasing.
    npoints: int, optional
        The number of linearly spaced points to generate the distribution function for. By default, 1000 points are used.
    boundary_value: unyt_quantity or float, optional
        The boundary value of the gravitational potential. If a float is provided, then the same unit assumptions are used
        as is done for the ``potential`` input. By default, the boundary value is 0.
    decreasing: bool, optional
        The ordering of ``potential``. If ``forward``, then it is assumed that the potential is monotonically increasing. Otherwise,
        it is assumed that the potential is monotonically decreasing.
    base_units: str or Unit, optional
        The base specific energy units to use. By default, the base units are :math:`{\rm cm^2\; s^{-2}}`.
    base_density_units: str or Unit, optional
        The base density units to use. By default, the base units are :math:`{\rm g/cm^3}`.

    Returns
    -------
    unyt_array
        The relative energy values (:math:`\mathcal{E}`) of the distribution function. These are length ``npoints`` and
        have units ``base_units``.
    unyt_array
        The distribution function values (:math:`f(\mathcal{E})`). These are also length ``npoints`` and have units
        of :math:`\left[\rho\right]\left[\mathcal{E}\right]^{-3/2}`.


    """
    # Validate units for the density and the potential. Either assume the units or enforce them via conversion.
    # Errors are allowed to stand because they are adequately explanatory.
    if hasattr(density, "units"):
        density = density.to_value(base_density_units)
    if hasattr(potential, "units"):
        potential = potential.to_value(base_units)
    density = np.asarray(density)
    potential = np.asarray(potential)

    # Construct the relative potential and check that it is increasing. We use the decreasing paramater to
    # determine which order to provide to the relative potential function.
    _order: Literal["forward", "backward"] = "forward" if decreasing else "backward"
    relpot = relative_potential(
        potential, boundary_value=boundary_value, order=_order, base_units=base_units
    ).d

    if relpot[0] > relpot[1]:
        raise ValueError(
            f"``relative_potential`` was computed in the {_order} direction, but is not monotonically increasing."
            "Switch the ``decreasing`` parameter to ensure that the relative potential is monotonically increasing."
        )

    # Construct the rho(psi) interpolator so that it can be used as part of the integration step.
    if _order == "backward":
        # We need to flip the density to ensure that everything is well behaved.
        density = density[::-1]

    rho_psi = InterpolatedUnivariateSpline(relpot, density)

    # Construct the integrand and integral functions. We use a transformation to omega = sqrt(E-psi) to
    # simplify the evaluation procedure. We first construct the integrand and then construct the actual
    # integral.
    integrand = lambda v, _rp: 2 * rho_psi(_rp - v**2, 1)
    integral = lambda _rp: quad(integrand, 0, np.sqrt(_rp), args=(_rp,))[0]

    # Construct relative energy array - we need to go from 0 to the maximum of the relative potential
    # and evaluate from there.
    Emin, Emax = 0, np.amax(relpot)
    Earr = np.linspace(Emin, Emax, npoints)

    # Perform the quadrature to obtain the values for each of the Earr values. Then enforce ordering and
    # the correct units.
    _f_int = np.asarray([integral(_E) for _E in Earr])
    fspline = InterpolatedUnivariateSpline(Earr, _f_int)
    f = (1 / np.sqrt(8) * np.pi**2) * fspline(Earr, 1)

    # Enforce the units convention for both Earr and f.
    Earr = unyt_array(Earr, base_units)
    f = unyt_array(f, Unit(base_units) ** (-3 / 2) * Unit(base_density_units))

    return Earr, f


def sample_eddington_distribution(
    relative_energies: Union[unyt_array, np.ndarray],
    df: Union[unyt_array, np.ndarray],
    particle_potentials: Union[unyt_array, np.ndarray],
    base_units: Union[str, Unit] = "cm**2/s**2",
    base_density_units: Union[str, Unit] = "g/cm**3",
    boundary_value: Union[unyt_quantity, float] = 0,
    **kwargs,
) -> unyt_array:
    r"""
    Sample 3D velocity vectors from an Eddington distribution for a set of particles
    at specified relative potentials.

    This function takes a precomputed Eddington distribution function :math:`f(\mathcal{E})`,
    tabulated over a range of relative energies (:math:`\mathcal{E}`), and uses it to
    generate velocity magnitudes consistent with each particle's local potential
    :math:`\Psi(r)`. It then assigns random 3D directions to produce isotropic velocities.

    Internally, it:
      1. Builds a spline from ``relative_energies``,``df``) using``InterpolatedUnivariateSpline``.
      2. Converts each particle's potential into a relative potential
         :math:`\mathrm{relpot} = -(\Phi - \Phi_0)`.
      3. Computes the local escape speed :math:`v_{\mathrm{esc}} = \sqrt{2 \,\mathrm{relpot}}`.
      4. Determines a bounding function for acceptance–rejection sampling,
         :math:`\mathrm{likelihood\_bound} = v_{\mathrm{esc}}^2 \times f(\mathrm{relpot})`.
      5. Calls a lower-level (Cython) function``generate_velocities`` to perform the
         actual acceptance–rejection sampling of velocity magnitudes.
      6. Draws random angles :math:`\theta, \phi` for each sampled magnitude to distribute
         the resulting velocity vectors isotropically in 3D.
      7. Returns the velocity vectors as an (N, 3) NumPy array.

    Parameters
    ----------
    relative_energies : np.ndarray or unyt_array, shape (M,)
        The sampled relative energies :math:`\mathcal{E}` used to define the distribution
        function, typically returned by``compute_eddington_distribution``. Must be sorted
        ascending.
    df : np.ndarray or unyt_array, shape (M,)
        The corresponding values of the Eddington DF, :math:`f(\mathcal{E})`, at each
        point in``relative_energies``. Must be positive in the range of interest (0 to max E).
        Units are typically :math:`[density]\,[\mathcal{E}]^{-3/2}`, consistent with the
        output of``compute_eddington_distribution``.
    particle_potentials : np.ndarray or unyt_array, shape (N,)
        The gravitational (or total) potential for each of N particles, which will be
        converted to a relative potential. For bound orbits, these values should be
        :math:`> 0` in relative units.
    base_units : str or Unit, optional
        The base energy units to assume for potentials and relative energies, by default
        'cm**2/s**2'.
    base_density_units : str or Unit, optional
        The density units associated with``df``, by default 'g/cm**3'.
    boundary_value : unyt_quantity or float, optional
        Reference potential value :math:`\Phi_0` used in computing the relative potential,
        by default 0.
    **kwargs
        Additional keyword arguments passed directly to the Cython function
       ``generate_velocities`` (e.g.,``show_progress=True``,``max_tries=50000``).

    Returns
    -------
    v : np.ndarray, shape (N, 3)
        The sampled 3D velocity vectors for each of the N particles. Magnitudes are drawn
        according to the Eddington DF, and directions are uniformly random over the sphere.

    Notes
    -----
    - The velocity magnitudes are sampled with an acceptance-rejection scheme specialized
      for the Eddington DF.
    - After magnitudes are sampled, directions are assigned by drawing angles uniformly:
      :math:`\phi \in [0, 2\pi`` and :math:``\theta \in [0, \pi]`.
    - If``particle_potentials`` or``relative_energies`` are given as unyt_arrays,
      they are converted internally to the specified base units.
    - The function returns a plain NumPy array for velocities. If you wish to reattach
      physical units (e.g., cm/s), you can do so afterward by noting that velocity is the
      square root of the potential’s unit.
    """
    # Create the spline for the eddington distribution so that
    # we can extract the relevant parameters to pass to the
    # cython level.
    if hasattr(relative_energies, "units"):
        relative_energies = relative_energies.to_value(base_units)
    if hasattr(particle_potentials, "units"):
        particle_potentials = particle_potentials.to_value(base_units)
    if hasattr(df, "units"):
        df = df.to_value(Unit(base_units) ** (-3 / 2) * Unit(base_density_units))

    df_spline = InterpolatedUnivariateSpline(relative_energies, df)
    t, c, k = df_spline.get_knots(), df_spline.get_coeffs(), df_spline._eval_args[2]

    # Construct the relative potential so that we have access to it.
    base_units = Unit(base_units)
    relpot = relative_potential(
        particle_potentials,
        base_units=base_units,
        order="forward",
        boundary_value=boundary_value,
    ).d
    vesc = np.sqrt(2 * relpot)
    likelihood_bound = vesc**2 * df_spline(relpot)

    # Pass to the cython level to sample the velocities.
    velocities = np.asarray(
        generate_velocities(relpot, vesc, likelihood_bound, t, c, k, **kwargs)
    )

    # Distribute the velocities randomly.
    _phi = np.random.uniform(0, 2 * np.pi, size=len(relative_energies))
    _theta = np.arccos(np.random.uniform(-1, 1, size=len(relative_energies)))
    v = np.stack(
        [
            velocities * np.sin(_theta) * np.cos(_phi),
            velocities * np.sin(_theta) * np.sin(_phi),
            velocities * np.cos(_theta),
        ],
        axis=-1,
    )

    return unyt_array(v, base_units ** (1 / 2))
