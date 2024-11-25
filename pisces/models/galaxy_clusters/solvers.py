from pisces.models.base import Solver
from pisces.models.utils import pipeline, state_checker
from typing import TYPE_CHECKING
import numpy as np
import unyt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from pisces.utilities.math import integrate,integrate_in_shells,integrate_toinf
from pisces.utilities.physics import mu,mue,m_p,G
from pisces.models.galaxy_clusters.static import STANDARD_FIELDS
if TYPE_CHECKING:
    from pisces._grids.grid_base import Grid


# noinspection PyMethodMayBeStatic
class DensityTotalDensitySolver(Solver):
    """
    Solver to compute fields based on density and total density, including dark matter content,
    mass fields, gravitational field, pressure, and temperature in a spherical geometry.
    """
    def _add_profiles_as_fields(self,grid):
        # This substep of the solver is applicable across all geometries and cases.
        # We are simply taking the profiles and dumping them to the grid.
        _profiles = ['density','total_density','stellar_density']
        _req = [True,True,False]

        for profile, is_required in zip(_profiles,_req):
            if is_required and profile not in self.model.profiles:
                raise ValueError(f"Missing required profile: '{profile}'.")

            if profile in self.model.profiles:
                self.add_field_from_profile(
                    grid,
                    profile,
                    units = STANDARD_FIELDS[profile]['units']
                )

    def _compute_dm_density_from_profiles(self,grid):
        total_density = grid.FIELDS['total_density'][...]
        combined_density = sum(
            grid.FIELDS[profile][...] if profile in grid.FIELDS else 0
            for profile in ['density', 'stellar_density']
        )
        dark_matter_density = total_density - combined_density

        grid.FIELDS.add_field(
            'dark_matter_density',
            data=dark_matter_density.d,
            units=str(dark_matter_density.units)
        )

    def _compute_mass_fields_spherical(self, grid: 'Grid'):
        # Grab the radii from the specific coordinate system for this model.
        radii = grid.get_coordinates().ravel() # Explicitly in kpc
        coordinate_system = self.model.geometry_handler.coordinate_system

        # Compute relevant integrals
        # start with profiles
        _profiles = ['total_density','density','stellar_density']
        _mass_field = ['total_mass','gas_mass','stellar_mass']
        for _p,_f in zip(_profiles,_mass_field):
            if _p in self.model.profiles:
                total = integrate_in_shells(
                self.model.profiles[_p], radii, coordinate_system=coordinate_system
                )
                grid.FIELDS.add_field(_f, data=total, units=STANDARD_FIELDS[_f]['units'])

        # Now integrate the dark matter via spline interpolation.
        dark_matter_spline = InterpolatedUnivariateSpline(
            radii, grid.FIELDS['dark_matter_density'][...].d
        )
        dark_matter_mass = integrate_in_shells(
            dark_matter_spline, radii, coordinate_system=coordinate_system
        )
        grid.FIELDS.add_field('dark_matter_mass', data=dark_matter_mass, units=STANDARD_FIELDS['dark_matter_mass']['units'])

    def _gravitational_field_spherical(self, grid: 'Grid'):
        radii = grid.get_coordinates().ravel()
        gravitational_field = -G * (grid.FIELDS['total_mass'][...] /
                                    unyt.unyt_array(radii, 'kpc')**2)
        grid.FIELDS.add_field(
            'gravitational_field',
            data=gravitational_field.to_value(STANDARD_FIELDS['gravitational_field']['units']),
            units=STANDARD_FIELDS['gravitational_field']['units']
        )

    def _compute_pressure_temperature_spherical(self, grid: 'Grid'):
        """
        Compute pressure and temperature fields.

        Parameters
        ----------
        grid : Grid
            The grid object where fields are added.
        """
        radii = grid.get_coordinates().ravel()
        gravitational_field = grid.FIELDS['gravitational_field'][...]
        density_profile = self.model.profiles['density']
        density = grid.FIELDS['density'][...]

        # Pressure Calculation
        g_spline = InterpolatedUnivariateSpline(radii, gravitational_field.d) # in standard units.
        dPdr = lambda r: density_profile(r) * g_spline(r)
        pressure = -integrate(dPdr, radii)

        # Add pressure correction for outer boundary
        outer_pressure_correction = lambda r: density_profile(r) * gravitational_field[-1] * (radii[-1] / r)**2
        pressure -= quad(outer_pressure_correction, radii[-1], np.inf, limit=100)[0]
        pressure = unyt.unyt_array(pressure,density.units*gravitational_field.units).to_value(STANDARD_FIELDS['pressure']['units'])

        grid.FIELDS.add_field('pressure', data=pressure, units=STANDARD_FIELDS['pressure']['units'])

        # Temperature Calculation
        temperature = pressure * mu * m_p / density
        temperature.convert_to_units(STANDARD_FIELDS['temperature']['units'])

        grid.FIELDS.add_field('temperature', data=temperature.d, units=STANDARD_FIELDS['temperature']['units'])

    @pipeline(geometry='SphericalCoordinateSystem', symmetry={'phi', 'theta'})
    def spherical_pipeline(self, grid: 'Grid'):
        """
        Pipeline for solving in spherical geometry with phi-theta symmetry.
        Adds fields to the grid, computes mass fields, gravitational fields, pressure,
        and temperature.
        """
        from pisces.utilities.math import integrate_in_shells, integrate
        from pisces.utilities.physics import G, mu, m_p

        # ---- Field Initialization ---- #
        # In this pipeline step, we pass all the relevant profiles to
        # the grid and then use arithemtic to determine the DM component.
        self._add_profiles_as_fields(grid)
        self._compute_dm_density_from_profiles(grid)

        # ---- Compute Mass Fields ---- #
        # We now integrate the mass components outward to
        # determine the total mass within the bound radius.
        self._compute_mass_fields_spherical(grid)

        # ---- Gravitational Field ---- #
        self._gravitational_field_spherical(grid)

        # ---- Pressure and Temperature ---- #
        self._compute_pressure_temperature_spherical(grid)

    @state_checker('geometry')
    def geometry_checker(self):
        """
        Validate the geometry of the coordinate system.
        """
        return self.model.geometry_handler.coordinate_system.__class__.__name__

    @state_checker('symmetry')
    def symmetry_checker(self):
        """
        Validate the symmetry axes of the coordinate system.
        """
        return {
            self.model.geometry_handler.coordinate_system.AXES[k]
            for k in self.model.geometry_handler.symmetry.symmetry_axes
        }
