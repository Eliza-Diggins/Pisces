from pathlib import Path
from typing import Union, List, Optional, Dict, TYPE_CHECKING

import unyt
from sympy.physics.units import femto

from pisces.utilities.math import integrate, integrate_toinf
from pisces.utilities.config import pisces_params
from pisces.utilities.physics import m_p, mu,G, mue
from numpy._typing import ArrayLike, NDArray
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from pisces.geometry import CoordinateSystem
from pisces.io import HDF5_File_Handle
from pisces.models.base import Model
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
from pisces.geometry import GeometryHandler, Symmetry
import numpy as np
from pisces.models.solver import solver_checker,solver_process
if TYPE_CHECKING:
    from pisces.profiles.base import Profile


class ClusterModel(Model):
    """
    A specialized model for spherical astrophysical clusters.

    This model supports spherical coordinate systems and pathways for
    creating profiles, calculating gravitational fields, and computing
    additional physical fields.

    Attributes
    ----------
    ALLOWED_COORDINATE_SYSTEMS : List[str]
        Allowed coordinate systems for the model.
    DEFAULT_COORDINATE_SYSTEMS : CoordinateSystem
        Default coordinate system used for the model.

    Notes
    -----
    The :py:class:`ClusterModel` can be implemented with any pseudo-spherical or spherical geometry. This
    includes :py:class:`~pisces.models.spherical_coordinates.SphericalCoordinateSystem` as well as other similar coordinate
    systems.
    """
    ALLOWED_COORDINATE_SYSTEMS = ['SphericalCoordinateSystem','OblateHomoeoidalCoordinateSystem']
    DEFAULT_COORDINATE_SYSTEMS = SphericalCoordinateSystem()

    @staticmethod
    def _correct_bbox(r_min: float, r_max: float) -> NDArray[np.floating]:
        """
        Construct the bounding box for the spherical coordinate system.

        Parameters
        ----------
        r_min : float
            Minimum radius of the grid.
        r_max : float
            Maximum radius of the grid.

        Returns
        -------
        NDArray[np.floating]
            Bounding box for the spherical coordinate system.

        Notes
        -----
        The bounding box is defined for the first octant of the spherical system.
        """
        return np.array([[r_min, 0, 0], [r_max, np.pi / 2, np.pi / 2]], dtype='f8')

    @staticmethod
    def _correct_scale(scale: Union[List[str], str]) -> List[str]:
        """
        Ensure the scale is appropriately formatted for spherical coordinates.

        Parameters
        ----------
        scale : Union[List[str], str]
            Scaling type for each axis (e.g., "linear" or "log").

        Returns
        -------
        List[str]
            Corrected scaling list for [r, theta, phi] axes.
        """
        if isinstance(scale, str):
            return [scale, 'linear', 'linear']
        return scale

    @classmethod
    def _correct_grid_shape(cls, grid_shape: Union[int, ArrayLike], coordinate_system: SphericalCoordinateSystem) -> \
    NDArray[np.int_]:
        """
        Validate and correct the grid shape for the model.

        Parameters
        ----------
        grid_shape : Union[int, ArrayLike]
            Shape of the grid.
        coordinate_system : SphericalCoordinateSystem
            Coordinate system for the model.

        Returns
        -------
        NDArray[np.int_]
            Corrected grid shape.

        Raises
        ------
        ValueError
            If the grid shape is invalid for the coordinate system.

        Notes
        -----
        If an integer is provided for `grid_shape`, it is expanded for spherical coordinates.
        """
        if isinstance(grid_shape, int):
            if coordinate_system.__class__.__name__ != 'SphericalCoordinateSystem':
                raise ValueError(
                    f"Integer 'grid_shape' is only valid for Spherical coordinates, not {coordinate_system}."
                )
            grid_shape = [grid_shape, 1, 1]
        return np.array(grid_shape, dtype='uint')

    # noinspection PyMethodOverriding
    @classmethod
    def build_skeleton(cls,
                       path: Union[str, Path],
                       r_min: float,
                       r_max: float,
                       grid_shape: Union[int, ArrayLike],
                       chunk_shape: Union[int, ArrayLike] = None,
                       *,
                       overwrite: bool = False,
                       length_unit: str = 'kpc',
                       scale: Union[List[str], str] = 'log',
                       profiles: Optional[Dict[str, 'Profile']] = None,
                       coordinate_system: Optional[SphericalCoordinateSystem] = None) -> 'HDF5_File_Handle':
        """
        Build the model skeleton for `ClusterModel`.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the model.
        r_min : float
            Minimum radius of the bounding box.
        r_max : float
            Maximum radius of the bounding box.
        grid_shape : Union[int, ArrayLike]
            Shape of the grid.
        chunk_shape : Union[int, ArrayLike], optional
            Shape of chunks in the grid.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
        length_unit : str, optional
            Unit of length for the grid. Default is 'kpc'.
        scale : Union[List[str], str], optional
            Scaling type for each axis. Default is 'log'.
        profiles : Optional[Dict[str, 'Profile']], optional
            Dictionary of profiles to include in the model.
        coordinate_system : Optional[SphericalCoordinateSystem], optional
            Coordinate system for the model.

        Returns
        -------
        HDF5_File_Handle
            Handle to the created HDF5 file.

        Notes
        -----
        This method initializes the model structure, grid, and profiles.
        """
        # MANAGE the coordinate system: we need to both check that
        # the coordinates are set and also validate them.
        if coordinate_system is None:
            coordinate_system = cls.DEFAULT_COORDINATE_SYSTEMS
        cls._cls_validate_coordinate_system(coordinate_system)

        # CORRECT the bounding box and the grid shape. This is necessary
        # because we always have the principle octant in these models, so we
        # don't need the user to tell us that.
        bbox = cls._correct_bbox(r_min, r_max)
        grid_shape = cls._correct_grid_shape(grid_shape, coordinate_system)
        if chunk_shape is not None:
            chunk_shape = cls._correct_grid_shape(chunk_shape, coordinate_system)
        scale = cls._correct_scale(scale)

        # PASS to the skeleton builder.
        return super().build_skeleton(
            path,
            bbox=bbox,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
            overwrite=overwrite,
            length_unit=length_unit,
            scale=scale,
            profiles=profiles,
            coordinate_system=coordinate_system,
        )

    # @@ BUILDERS @@ #
    @classmethod
    def from_dens_and_tden(cls,
                           path: str,
                           r_min: float,
                           r_max: float,
                           grid_shape: Union[int, ArrayLike],
                           density: 'Profile',
                           total_density: 'Profile',
                           chunk_shape: Union[int, ArrayLike] = None,
                           extra_profiles: Optional[Dict[str, 'Profile']] = None,
                           coordinate_system: Optional[CoordinateSystem] = None,
                           **kwargs) -> 'ClusterModel':
        """
        Create a ClusterModel from density and total density profiles.

        Parameters
        ----------
        path : str
            Path to save the model.
        r_min : float
            Minimum radius.
        r_max : float
            Maximum radius.
        grid_shape : Union[int, ArrayLike]
            Shape of the grid.
        density : Profile
            Profile for density.
        total_density : Profile
            Profile for total density.
        chunk_shape : Union[int, ArrayLike], optional
            Shape of chunks in the grid.
        extra_profiles : Optional[Dict[str, Profile]], optional
            Additional profiles to include.
        coordinate_system : Optional[CoordinateSystem], optional
            Coordinate system for the model.

        Returns
        -------
        ClusterModel
            Constructed ClusterModel instance.
        """
        # @@ CONSTRUCT SKELETON @@ #
        # This generates the necessary file structure and background
        # data for the model. All of the structure generation should
        # be the same between coordinate systems.
        profiles = {'density': density, 'total_density': total_density}
        if extra_profiles:
            profiles.update(extra_profiles)

        # build the skeleton for the system and initialize
        # the model object.
        cls.build_skeleton(
            path,
            r_min=r_min,
            r_max=r_max,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
            coordinate_system=coordinate_system,
            profiles=profiles,
            **kwargs,
        )
        obj = cls(path)

        # @@ RUN MODEL PATHWAY @@ #
        # This now triggers the physics computations to generate
        # the full model.
        if obj.coordinate_system.__class__.__name__ == 'SphericalCoordinateSystem':
            obj(pathway='spherical_dens_tden')
        elif 'Homoeoidal' in obj.coordinate_system.__class__.__name__:
            obj(pathway='homoeoidal_dens_tden')
        else:
            raise NotImplementedError(f"The coordinate system {obj.coordinate_system.__class__.__name__} is an accepted coordinate"
                                      f" system for {cls.__name__}, but there is no density / total density pipeline implemented.")

        return obj

    @classmethod
    def from_dens_and_temp(cls,
                           path: str,
                           r_min: float,
                           r_max: float,
                           grid_shape: Union[int, ArrayLike],
                           density: 'Profile',
                           temperature: 'Profile',
                           chunk_shape: Union[int, ArrayLike] = None,
                           extra_profiles: Optional[Dict[str, 'Profile']] = None,
                           coordinate_system: Optional[CoordinateSystem] = None,
                           **kwargs) -> 'ClusterModel':
        """
        Create a ClusterModel from density and total density profiles.

        Parameters
        ----------
        path : str
            Path to save the model.
        r_min : float
            Minimum radius.
        r_max : float
            Maximum radius.
        grid_shape : Union[int, ArrayLike]
            Shape of the grid.
        density : Profile
            Profile for density.
        temperature : Profile
            Profile for the ICM temperature.
        chunk_shape : Union[int, ArrayLike], optional
            Shape of chunks in the grid.
        extra_profiles : Optional[Dict[str, Profile]], optional
            Additional profiles to include.
        coordinate_system : Optional[CoordinateSystem], optional
            Coordinate system for the model.

        Returns
        -------
        ClusterModel
            Constructed ClusterModel instance.
        """
        # @@ CONSTRUCT SKELETON @@ #
        # This generates the necessary file structure and background
        # data for the model. All of the structure generation should
        # be the same between coordinate systems.
        profiles = {'density': density, 'temperature': temperature}
        if extra_profiles:
            profiles.update(extra_profiles)

        # build the skeleton for the system and initialize
        # the model object.
        cls.build_skeleton(
            path,
            r_min=r_min,
            r_max=r_max,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
            coordinate_system=coordinate_system,
            profiles=profiles,
            **kwargs,
        )
        obj = cls(path)

        # @@ RUN MODEL PATHWAY @@ #
        # This now triggers the physics computations to generate
        # the full model.
        if obj.coordinate_system.__class__.__name__ == 'SphericalCoordinateSystem':
            obj(pathway='spherical_dens_temp')
        elif 'Homoeoidal' in obj.coordinate_system.__class__.__name__:
            obj(pathway='homoeoidal_dens_temp')
        else:
            raise NotImplementedError(f"The coordinate system {obj.coordinate_system.__class__.__name__} is an accepted coordinate"
                                      f" system for {cls.__name__}, but there is no density / temperature pipeline implemented.")

        return obj

    # @@ CHECKERS @@ #
    @solver_checker('spherical_dens_temp')
    @solver_checker('spherical_dens_tden')
    @solver_checker('homoeoidal_dens_tden')
    @solver_checker('homoeoidal_dens_temp')
    def _check_pathways(self,pathway: str):
        # CHECKING geometry
        state = True
        if pathway.startswith('spherical'):
            state = state and (self.coordinate_system.__class__.__name__ == 'SphericalCoordinateSystem')
        if pathway.startswith('homoeidal'):
            state = state and ('Homoeoidal' in self.coordinate_system.__class__.__name__)

        # CHECKING profiles
        if pathway.endswith("dens_temp"):
            state = state and ('temperature' in self.profiles)
            state = state and ('density' in self.profiles)
        elif pathway.endswith("dens_tden"):
            state = state and ('total_density' in self.profiles)
            state = state and ('density' in self.profiles)

        return state


    # @@ UTILITY METHODS @@ #
    # These are generic methods which contain processes which occur in
    # various contexts within the ClusterModel class.
    @solver_process('spherical_dens_temp', step=0)
    @solver_process('spherical_dens_tden', step=0)
    @solver_process('homoeoidal_dens_temp', step=0)
    @solver_process('homoeoidal_dens_tden', step=0)
    def _dump_profiles_to_fields(self):
        """
        Transfer profiles into the model's fields.

        Adds fields for all profiles stored in the model, ensuring that
        the `stellar_density` field is also initialized if missing.
        """
        for prof in self.profiles.keys():
            self.add_field_from_profile(
                profile_name=prof,
                chunking=False,
                units=pisces_params[f'fields.gclstr.{prof}.units'],
            )

        if 'stellar_density' not in self.FIELDS:
            self.logger.info("[SLVR] Adding null stellar density field 'stellar_density'.")
            self.FIELDS.add_field(
                'stellar_density',
                axes=['r'],
                units=pisces_params['fields.gclstr.stellar_density.units'],
            )

    @solver_process('spherical_dens_tden', step=1)
    @solver_process('homoeoidal_dens_tden', step=1)
    @solver_process('spherical_dens_temp', step=5)
    def _compute_dm_density(self):
        """
        Compute the dark matter density field.

        Calculates the `dark_matter_density` field by subtracting the sum
        of the stellar density and gas density from the total density.
        """
        total_density = self.FIELDS['total_density'][...]
        stellar_density = self.FIELDS['stellar_density'][...]
        gas_density = self.FIELDS['density'][...]

        dark_matter_density = total_density - (stellar_density + gas_density)
        self.FIELDS.add_field(
            'dark_matter_density',
            axes=['r'],
            data=dark_matter_density.to_value(pisces_params[f'fields.gclstr.dark_matter_density.units']),
            units=pisces_params['fields.gclstr.dark_matter_density.units'],
        )
        self.logger.debug("Field 'dark_matter_density' added.")

    @solver_process('spherical_dens_tden', step=2, kwargs=dict(fields=None))
    @solver_process('homoeoidal_dens_tden', step=2, kwargs=dict(fields=None))
    @solver_process('spherical_dens_temp', step=6)
    def _get_masses_from_density(self, fields=None):
        """
        Compute mass fields from density fields.

        Parameters
        ----------
        fields : List[str], optional
            List of density field names to integrate. Defaults to all relevant fields.

        Integrates each density field over spherical shells to compute
        corresponding mass fields and adds them to the model's fields.
        """
        fields = fields or ['total_density', 'density', 'stellar_density', 'dark_matter_density']
        radius = self.grid_manager.get_coordinates(axes=['r']).ravel()

        for field in fields:
            if field in self.profiles:
                density_function = self.profiles[field]
                density_units = density_function.units
            elif field in self.FIELDS:
                density_data = self.FIELDS[field][...].d
                density_function = InterpolatedUnivariateSpline(radius, density_data)
                density_units = self.FIELDS[field].units
            else:
                raise ValueError(f"Density field '{field}' not found in FIELDS or profiles.")

            shell_masses = self.coordinate_system.integrate_in_shells(radius, density_function)
            mass_units = unyt.Unit(density_units) * unyt.Unit(self.grid_manager.length_unit) ** 3
            mass_field = pisces_params[f'fields.gclstr.{field}.mass_field']
            target_units = pisces_params[f'fields.gclstr.{mass_field}.units']
            shell_masses = unyt.unyt_array(shell_masses, mass_units).to(target_units)

            self.FIELDS.add_field(
                mass_field,
                axes=['r'],
                units=shell_masses.units,
                data=shell_masses.d,
            )
            self.logger.debug(f"Field '{mass_field}' added.")

    @solver_process('spherical_dens_tden', step=3)
    @solver_process('spherical_dens_temp', step=3,kwargs=dict(out='total_mass'))
    def _compute_spherical_gfield(self,out='gravitational_field'):
        """
        Compute the gravitational field using Gauss's theorem.

        Computes the radial gravitational field from the total mass profile
        and radius.
        """
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        if out == 'gravitational_field':
            total_mass = self.FIELDS['total_mass'][...]

            field = (-total_mass * G) / (radius ** 2)
            field = field.to(pisces_params['fields.gclstr.gravitational_field.units'])
        elif out == 'total_mass':
            gravitational_field = self.FIELDS['gravitational_field'][...]
            field = (radius**2)*(-gravitational_field)/G
            field = field.to(pisces_params['fields.gclstr.total_mass.units'])
        else:
            raise ValueError(f"Output '{out}' not recognized.")

        self.FIELDS.add_field(
            out,
            axes=['r'],
            units=field.units,
            data=field.d,
        )
        self.logger.debug(f"Field '{out}' added.")

    @solver_process('spherical_dens_temp', step=4)
    def _compute_gas_total_density(self):
        # PULL fields
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        total_mass = self.FIELDS['total_mass'][...]

        # INTERPOLATE
        tmspline = InterpolatedUnivariateSpline(radius.d,total_mass.d)

        # GRAB the volume elements
        shell_volume = self.coordinate_system.shell_volume(radius.d)
        total_density = tmspline(radius.d,1)/shell_volume
        td_unit = total_mass.units/(radius.units**3)
        total_density = unyt.unyt_array(total_density,td_unit)
        total_density = total_density.to(pisces_params['fields.gclstr.total_density.units'])

        self.FIELDS.add_field(
            'total_density',
            axes=['r'],
            units=total_density.units,
            data=total_density.d,
        )

    @solver_process('spherical_dens_tden', step=4)
    def _compute_pressure_from_g(self):
        """
        Compute the pressure field using the gravitational field.

        Integrates the product of density and gravitational field to
        determine pressure.
        """
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        gravitational_field = self.FIELDS['gravitational_field'][...]
        density = self.profiles['density']

        gspline = InterpolatedUnivariateSpline(radius.d, gravitational_field.d)

        integrand_inner = lambda r: gspline(r) * density(r)
        integrand_outer = lambda r: density(r) * gravitational_field.d[-1] * (radius.d[-1] / r) ** 2

        pressure = -integrate(integrand_inner, radius.d, x_0=radius.d[-1])
        pressure -= quad(integrand_outer, radius.d[-1], np.inf)[0]

        pressure_units = radius.units * gravitational_field.units * density.units
        pressure = unyt.unyt_array(pressure, pressure_units)

        self.FIELDS.add_field(
            'pressure',
            axes=['r'],
            units=pressure.units,
            data=pressure.d,
        )
        self.logger.debug("Field 'pressure' added.")

    @solver_process('spherical_dens_temp', step=2)
    @solver_process('homoeoidal_dens_temp', step=2)
    def _compute_g_from_pressure(self):
        # EXTRACT the necessary fields from the coordinate system.
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        pressure = self.FIELDS['pressure'][...]
        density = self.FIELDS['density'][...]

        # CONSTRUCT the relevant geometry manager.
        symmetry = Symmetry(['theta','phi'],self.coordinate_system)
        handler = GeometryHandler(self.coordinate_system,symmetry)
        grad_symmetry = symmetry.gradient_component(0,basis='covariant')
        coordinates = self.grid_manager.get_coordinates(axes=grad_symmetry.get_asymmetric_coord_axes())

        # CONSTRUCT the pressure spline and its derivative function.
        # Then proceed to compute the gradient of the pressure.
        pspline = InterpolatedUnivariateSpline(radius.d, pressure.d)
        dpsline = lambda coords: pspline(coords[...,0],1)
        grad_p = handler.compute_function_gradient_term(coordinates,pspline,0,derivative=dpsline)

        _gravitational_field = grad_p/density.d
        _gravitational_field = unyt.unyt_array(_gravitational_field, pressure.units/(density.units*radius.units))
        _gravitational_field = _gravitational_field.to(pisces_params['fields.gclstr.gravitational_field.units'])

        self.FIELDS.add_field(
            'gravitational_field',
            axes=grad_symmetry.get_asymmetric_coord_axes(),
            units=_gravitational_field.units,
            data=_gravitational_field.d,
        )
        self.logger.debug("Field 'gravitational_field' added.")

    @solver_process('homoeoidal_dens_temp', step=3)
    @solver_process('spherical_dens_temp', step=8)
    def _compute_phi_from_pressure(self):
        # EXTRACT the necessary fields from the coordinate system.
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        pressure = self.FIELDS['pressure'][...]
        density = self.FIELDS['density'][...]

        # CONSTRUCT splines
        pspline = InterpolatedUnivariateSpline(radius.d, pressure.d)
        dspline = InterpolatedUnivariateSpline(radius.d, density.d)

        integrand = lambda r: pspline(r,1)/dspline(r)

        # PERFORM the integration
        _phi = integrate_toinf(integrand,radius.d)
        _phi = unyt.unyt_array(_phi, radius.units*pressure.units/density.units)

        self.FIELDS.add_field(
            'gravitational_potential',
            axes=['r'],
            units=_phi.units,
            data=_phi.d,
        )
        self.logger.debug("Field 'gravitational_potential' added.")

    @solver_process('homoeoidal_dens_temp', step=4)
    def _compute_dyn_density_from_phi(self):
        # EXTRACT the necessary fields
        radius = unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                                 self.grid_manager.length_unit)
        potential = self.FIELDS['gravitational_potential'][...]

        # Get the grid of coordinates
        coordinates = self.grid_manager.get_coordinates(axes=['r','theta'])

        # construct the manager
        symmetry = Symmetry(['theta','phi'],self.coordinate_system)
        handler = GeometryHandler(self.coordinate_system,symmetry)
        # construct the splines
        pspline = InterpolatedUnivariateSpline(radius.d, potential.d)

        lap_potential = handler.function_laplacian(coordinates,pspline,active_axes=[0])



    @solver_process('spherical_dens_temp',step=1,kwargs=dict(out='pressure'))
    @solver_process('homoeoidal_dens_temp',step=1,kwargs=dict(out='pressure'))
    @solver_process('spherical_dens_tden', step=5)
    def _compute_from_eos(self, out='temperature'):
        """
        Compute a field from the equation of state.

        Parameters
        ----------
        out : str
            Output field to compute ('temperature', 'pressure', or 'density').
        """
        scale_factor = m_p * mu

        if out == 'temperature':
            density = self.FIELDS['density'][...]
            pressure = self.FIELDS['pressure'][...]
            field = (pressure * scale_factor) / density
        elif out == 'pressure':
            density = self.FIELDS['density'][...]
            temperature = self.FIELDS['temperature'][...]
            field = (temperature * density) / scale_factor
        elif out == 'density':
            pressure = self.FIELDS['pressure'][...]
            temperature = self.FIELDS['temperature'][...]
            field = (pressure * scale_factor) / temperature
        else:
            raise ValueError(f"Output '{out}' not recognized.")

        field = field.to(pisces_params[f'fields.gclstr.{out}.units'])
        self.FIELDS.add_field(
            out,
            axes=['r'],
            units=field.units,
            data=field.d,
        )
        self.logger.debug(f"Field '{out}' added.")

    @solver_process('spherical_dens_tden', step=6)
    @solver_process('spherical_dens_temp', step=7)
    def _compute_auxiliary_fields(self):
        """
        Compute auxiliary fields like gas fraction, electron density, and entropy.
        """
        density = self.FIELDS['density'][...]
        total_density = self.FIELDS['total_density'][...]
        gas_fraction = density / total_density

        self.FIELDS.add_field(
            'gas_fraction',
            axes=['r'],
            units='',
            data=gas_fraction.d,
        )

        electron_density = density.to("cm**-3", "number_density", mu=mue)
        self.FIELDS.add_field(
            'electron_number_density',
            axes=['r'],
            units=electron_density.units,
            data=electron_density.d,
        )

        entropy = self.FIELDS['temperature'][...] * self.FIELDS['electron_number_density'][...] ** (-2 / 3)
        self.FIELDS.add_field(
            'entropy',
            axes=['r'],
            units=entropy.units,
            data=entropy.d,
        )



if __name__ == '__main__':
    from pisces.profiles import NFWDensityProfile, IsothermalTemperatureProfile
    from pisces.geometry import SphericalCoordinateSystem, OblateHomoeoidalCoordinateSystem
    d = NFWDensityProfile(rho_0=1e5,r_s=100)
    td = NFWDensityProfile(rho_0=1e6,r_s=150)
    t = IsothermalTemperatureProfile(T_0=5)

    cs = SphericalCoordinateSystem()
    cs_ho = OblateHomoeoidalCoordinateSystem(ecc=0.5)

    model_hom = ClusterModel.from_dens_and_temp('test_hom.hdf5',1e-1,1e4,[500,500,500],d,t,coordinate_system=cs_ho,overwrite=True)
    model = ClusterModel.from_dens_and_temp('test.hdf5',1e-1,1e4,[500,500,500],d,t,coordinate_system=cs,overwrite=True)

    import matplotlib.pyplot as plt

    radii_hom = model_hom.grid_manager.get_coordinates(axes=['r']).ravel()
    radii_sph = model.grid_manager.get_coordinates(axes=['r']).ravel()
    total_mass_sph = model.FIELDS['gravitational_potential'][...]
    total_mass_hom = model_hom.FIELDS['gravitational_potential'][...]
#
    plt.loglog(radii_hom, -total_mass_hom)
    plt.loglog(radii_sph, -total_mass_sph)
    plt.show()
