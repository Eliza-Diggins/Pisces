from pathlib import Path
from typing import Union, List, Optional, Dict, TYPE_CHECKING

import numpy as np
import unyt
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import make_interp_spline
from pisces.utilities.array_utils import make_grid_fields_broadcastable
from pisces.geometry import CoordinateSystem
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
from pisces.io import HDF5_File_Handle
from pisces.models.base import Model
from pisces.models.galaxy_clusters.grids import ClusterGridManager
from pisces.models.galaxy_clusters.utils import gcluster_params
from pisces.models.solver import solver_checker, solver_process
from pisces.utilities.math_utils.numeric import integrate_vectorized
from pisces.utilities.physics import m_p, mu, G
from utilities.math_utils.numeric import integrate

if TYPE_CHECKING:
    from pisces.profiles.base import Profile


class ClusterModel(Model):
    r"""
    A specialized model class for idealized galaxy clusters.

    Attributes
    ----------
    ALLOWED_COORDINATE_SYSTEMS : List[str]
        Allowed coordinate systems for the model.
    DEFAULT_COORDINATE_SYSTEM : CoordinateSystem
        Default coordinate system used for the model.

    Notes
    -----
    The :py:class:`ClusterModel` can be implemented with any pseudo-spherical or spherical geometry. This
    includes :py:class:`~pisces.models.spherical_coordinates.SphericalCoordinateSystem` as well as other similar coordinate
    systems.
    """
    # @@ VALIDATION MARKERS @@ #
    # Galaxy cluster models permit any subset of pseudo-spherical coordinates.
    # TODO: In the long run, this should really support most radial coordinate systems.
    ALLOWED_COORDINATE_SYSTEMS = ['SphericalCoordinateSystem',
                                  'OblateHomoeoidalCoordinateSystem',
                                  'ProlateHomoeoidalCoordinateSystem']

    # @@ CLASS PARAMETERS @@ #
    # The only meaningful difference here is that a custom GridManager is used
    # for validation and INIT_FREE_AXES starts with everything being radial.
    DEFAULT_COORDINATE_SYSTEM = SphericalCoordinateSystem
    GRID_MANAGER_CLASS = ClusterGridManager
    INIT_FREE_AXES = ['r']

    # @@ UTILITY METHODS @@ #
    @classmethod
    def get_default_units(cls,field_name: str) -> unyt.Unit:
        """
        Retrieve the default units for a specified field.

        This method looks up the default units for a given field name from the
        :py:attr:`~pisces.models.galaxy_clusters.utils.gcluster_params` configuration
        dictionary, which stores model-specific parameters including field units.
        If the field name is not found in the configuration, an exception is raised.

        Parameters
        ----------
        field_name : str
            The name of the field for which the default units are to be retrieved.

        Returns
        -------
        unyt.Unit
            The corresponding units for the specified field as a `unyt.Unit` object.

        Raises
        ------
        ValueError
            If the specified field name is not found in the :py:attr:`~pisces.models.galaxy_clusters.utils.gcluster_params`
            dictionary.

        Examples
        --------

        Let's fetch the default temperature unit:

        >>> ClusterModel.get_default_units("temperature")
        keV

        """
        # Fetch the unit data from the field entry in the
        # parameters object.
        try:
            _unit_str = gcluster_params[f'fields.{field_name}.units']
            if _unit_str is None:
                raise KeyError()
        except KeyError:
            raise ValueError("Failed to get default units for field `%s` because it is not a known field."%field_name)

        # Attempt to return the unit
        try:
            return unyt.Unit(_unit_str)
        except Exception as e:
            raise ValueError(f"Failed to convert string unit {_unit_str} to unyt.Unit object: {e}")

    # @@ CONSTRUCTION METHODS @@ #
    # `build_skeleton` gets overwritten here because we want to implement the gcluster
    # specific norms for the bbox and grid spacing.
    # noinspection PyMethodOverriding
    @classmethod
    def build_skeleton(cls,
                       path: Union[str, Path],
                       r_min: float,
                       r_max: float,
                       /,
                       num_points: Optional[int] = 1000,
                       n_phi: Optional[int] = None,
                       n_theta: Optional[int] = None,
                       chunk_shape: Union[int, ArrayLike] = None,
                       *,
                       overwrite: bool = False,
                       length_unit: str = 'kpc',
                       scale: Union[List[str], str] = 'log',
                       profiles: Optional[Dict[str, 'Profile']] = None,
                       coordinate_system: Optional[SphericalCoordinateSystem] = None) -> 'HDF5_File_Handle':
        r"""
        Build the skeleton for initializing a new model.

        This method provides the basic HDF5 structure in ``path`` to initialize it as a
        :py:class:`Model` instance. This includes managing the profiles (``profiles``), the coordinate
        system (``coordinate_system``), and the grid shape / boundaries.

        Parameters
        ----------
        path : Union[str, Path]
            Path at which to create the model skeleton. If the path already exists,
            ``overwrite`` determines the behavior.
        r_min: float
            The minimum radius of the galaxy cluster in units of ``length_unit``.
        r_max: float
            The maximum radius of the galaxy cluster in units of ``length_unit``.
        num_points: int, optional
            The number of radial points in the galaxy cluster's grid. By default, ``num_points = 1000``.
        n_phi: int, optional
            The number of grid points to place in the :math:`\phi` axis. This attribute is **only required** when
            using a triaxial coordinate system; otherwise, it is redundant and is unused.
        n_theta: int, optional
            The number of grid points to place in the :math:`\theta` axis. This attribute is **only required** when
            using a non-spherical coordinate system; otherwise, it is redundant and is unused.
        chunk_shape : ArrayLike, optional
            Shape of chunks in the grid. The ``chunk_shape`` should follow the same conventions as ``grid_shape``;
            however, it must also be a whole factor of ``grid_shape`` (``grid_shape % chunk_shape == 0``). In some
            instances, operations may be performed in chunks instead of on the entire grid at once. In these cases,
            the chunk shape balances efficient computation with memory consumption.
        overwrite : bool, optional
            If True, overwrite any existing file at the specified path. If False and the
            file exists, an exception will be raised. Defaults to False.
        length_unit : str, optional
            The unit of measurement for grid lengths (e.g., 'kpc', 'Mpc'). Defaults to 'kpc'.
        scale : Union[List[str], str], optional
            The scaling type for each grid axis. Accepted values are 'linear' or 'log'. If
            a single value is provided, it is applied to all axes. Defaults to 'linear'.
        profiles : Optional[Dict[str, 'Profile']], optional
            A dictionary of profiles to initialize in the model. Each key-value pair
            represents the profile name and corresponding `Profile` object.
        coordinate_system : Optional[CoordinateSystem], optional
            The coordinate system for the model. If None, the `DEFAULT_COORDINATE_SYSTEM`
            of the class is used.

        Returns
        -------
        HDF5_File_Handle
            The HDF5 file handle.

        Raises
        ------
        ValueError
            If required parameters are missing or validation fails.

        Notes
        -----

        """
        # Initialize the coordinate system including the default implementation if the
        # coordinate system isn't actually provided. NOTE: we don't differ to super() here because
        # we need to validate the coordinate system anyway.
        if coordinate_system is None:
            coordinate_system = cls.DEFAULT_COORDINATE_SYSTEM(**cls.DEFAULT_COORDINATE_SYSTEM_PARAMS)
        cls._cls_validate_coordinate_system(coordinate_system)

        # Determine requirements based on coordinate system
        requires_theta = coordinate_system.__class__.__name__ not in ['SphericalCoordinateSystem']
        requires_phi = coordinate_system.__class__.__name__ not in ['OblateHomoeoidalCoordinateSystem',
                                                                  'ProlateHomoeoidalCoordinateSystem',
                                                                    'SphericalCoordinateSystem']

        # Ensure that the user has provided n_phi and n_theta if they are required.
        # Otherwise, we return an error.
        n_phi = n_phi or (1 if not requires_phi else None)
        if n_phi is None:
            raise ValueError(f"Parameter `n_phi` is required for coordinate systems "
                             f"of type {coordinate_system.__class__.__name__}.")
        n_theta = n_theta or (1 if not requires_theta else None)
        if n_theta is None:
            raise ValueError(f"Parameter `n_theta` is required for coordinate systems "
                             f"of type {coordinate_system.__class__.__name__}.")

        # Utilize the custom validation mode in ClusterGridManager to validate / refactor
        # the bbox, scale, and grid_shape. Then pass to the parent method.
        grid_shape = [num_points, n_theta, n_phi]
        scale = cls.GRID_MANAGER_CLASS.correct_scale(scale)
        bbox = cls.GRID_MANAGER_CLASS.correct_bbox(r_min, r_max)

        # Call parent skeleton builder
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
                           gas_density: 'Profile',
                           total_density: 'Profile',
                           num_points: Optional[int] = 1000,
                           n_phi: Optional[float] = None,
                           n_theta: Optional[float] = None,
                           chunk_shape: Union[int, ArrayLike] = None,
                           extra_profiles: Optional[Dict[str, 'Profile']] = None,
                           coordinate_system: Optional[CoordinateSystem] = None,
                           **kwargs) -> 'ClusterModel':
        r"""
        Construct a `ClusterModel` from density and total density profiles.

        This method initializes the galaxy cluster model using two profiles:
        `density` and `total_density`. It sets up the grid, initializes the profiles,
        and triggers the appropriate solver pathway to compute the physical fields.

        Parameters
        ----------
        path : str
            Path to the HDF5 file where the model will be created. If the file exists and
            `overwrite` is not set to True, an exception will be raised.
        r_min : float
            Minimum radius of the model grid in units specified by `length_unit`.
        r_max : float
            Maximum radius of the model grid in units specified by `length_unit`.
        gas_density : Profile
            The gas / ICM density profile of the cluster as a function of the radius.
        total_density : Profile
            The dynamical density profile of the cluster as a function of the radius.
        num_points : int, optional
            Number of grid points along the radial (r) axis. Defaults to 1000.
        n_phi : float, optional
            Number of grid points along the azimuthal (:math:`\phi`) axis. This is only required
            when using a triaxial coordinate system. Defaults to None.
        n_theta : float, optional
            Number of grid points along the polar (:math:`\theta`) axis. This is required
            for non-spherical coordinate systems. Defaults to None.
        chunk_shape : Union[int, ArrayLike], optional
            The shape of chunks used to store grid data in the HDF5 file. Chunks allow efficient
            I/O operations for large datasets. Defaults to None.
        extra_profiles : Optional[Dict[str, 'Profile']], optional
            Additional profiles to include in the model. These are passed as a dictionary
            where keys are profile names and values are `Profile` objects. Defaults to None.
        coordinate_system : Optional[CoordinateSystem], optional
            The coordinate system for the model grid. If None, the default coordinate system
            defined by `DEFAULT_COORDINATE_SYSTEM` is used.
        **kwargs
            Additional keyword arguments passed to the `build_skeleton` method.

        Returns
        -------
        ClusterModel
            An initialized instance of the `ClusterModel` class with the skeleton created
            and physical fields computed from the density and total density profiles.

        Raises
        ------
        NotImplementedError
            If the specified coordinate system does not have a corresponding solver pathway
            for the provided profiles.

        Notes
        -----

        .. image:: ../diagrams/gclstr_dens_temp_sphere.png


        """
        # Set up the profiles. These get fed into build_skeleton and then
        # dumped to fields as the first step in the pipeline.
        profiles = {'gas_density': gas_density,
                    'total_density': total_density}
        if extra_profiles:
            profiles.update(extra_profiles)

        # build the skeleton for the system and initialize
        # the model object.
        cls.build_skeleton(
            path,
            r_min,
            r_max,
            num_points=num_points,
            n_phi=n_phi,
            n_theta=n_theta,
            chunk_shape=chunk_shape,
            coordinate_system=coordinate_system,
            profiles=profiles,
            **kwargs,
        )
        obj = cls(path)

        # Run the solver on the generated object to solve for the relevant fields.
        coordinate_system_name = obj.coordinate_system.__class__.__name__
        if coordinate_system_name == 'SphericalCoordinateSystem':
            obj(pathway='spherical_dens_tden')
        elif 'Homoeoidal' in coordinate_system_name:
            obj(pathway='homoeoidal_dens_tden')
        else:
            raise NotImplementedError(f"The coordinate system {coordinate_system_name} is an accepted coordinate"
                                      f" system for {cls.__name__}, but there is no density / total density pipeline implemented.")

        return obj

    @classmethod
    def from_dens_and_temp(cls,
                           path: str,
                           r_min: float,
                           r_max: float,
                           /,
                           gas_density: 'Profile',
                           temperature: 'Profile',
                           *,
                           num_points: Optional[int] = 1000,
                           n_phi: Optional[float] = None,
                           n_theta: Optional[float] = None,
                           chunk_shape: Union[int, ArrayLike] = None,
                           extra_profiles: Optional[Dict[str, 'Profile']] = None,
                           coordinate_system: Optional[CoordinateSystem] = None,
                           **kwargs) -> 'ClusterModel':
        r"""
        Construct a `ClusterModel` from gas density and temperature profiles.

        This method initializes the galaxy cluster model using two profiles:
        `density` and `total_density`. It sets up the grid, initializes the profiles,
        and triggers the appropriate solver pathway to compute the physical fields.

        Parameters
        ----------
        path : str
            Path to the HDF5 file where the model will be created. If the file exists and
            `overwrite` is not set to True, an exception will be raised.
        r_min : float
            Minimum radius of the model grid in units specified by `length_unit`.
        r_max : float
            Maximum radius of the model grid in units specified by `length_unit`.
        gas_density : Profile
            The gas / ICM density profile of the cluster as a function of the radius.
        temperature : Profile
            The temperature profile of the cluster as a function of the radius.
        num_points : int, optional
            Number of grid points along the radial (r) axis. Defaults to 1000.
        n_phi : float, optional
            Number of grid points along the azimuthal (:math:`\phi`) axis. This is only required
            when using a triaxial coordinate system. Defaults to None.
        n_theta : float, optional
            Number of grid points along the polar (:math:`\theta`) axis. This is required
            for non-spherical coordinate systems. Defaults to None.
        chunk_shape : Union[int, ArrayLike], optional
            The shape of chunks used to store grid data in the HDF5 file. Chunks allow efficient
            I/O operations for large datasets. Defaults to None.
        extra_profiles : Optional[Dict[str, 'Profile']], optional
            Additional profiles to include in the model. These are passed as a dictionary
            where keys are profile names and values are `Profile` objects. Defaults to None.
        coordinate_system : Optional[CoordinateSystem], optional
            The coordinate system for the model grid. If None, the default coordinate system
            defined by ``DEFAULT_COORDINATE_SYSTEM`` is used.
        **kwargs
            Additional keyword arguments passed to the `build_skeleton` method.

        Returns
        -------
        ClusterModel
            An initialized instance of the :py:class:`ClusterModel` class with the skeleton created
            and physical fields computed from the density and total density profiles.

        Raises
        ------
        NotImplementedError
            If the specified coordinate system does not have a corresponding solver pathway
            for the provided profiles.

        Notes
        -----
        The :py:meth:`ClusterModel.from_dens_and_temp` method provides direct access to two separate construction
        pathways:

        - ``"spherical_dens_temp"``: When the coordinate system is :py:class:`~pisces.geometry.coordinate_systems.SphericalCoordinateSystem`.

        - ``"homoeoidal_dens_temp"``: When the coordinate system is homoeoidal (either :py:class:`~pisces.geometry.coordinate_systems.ProlateHomoeoidalCoordinateSystem` or :py:class:`~pisces.geometry.coordinate_systems.OblateHomoeoidalCoordinateSystem`).

        In their implementation, both pathways are quite similar; however, special allowances have to be made for the
        geometric difficulties which arise from non-spherical geometry. The pathway diagrams for each are as follows:

        .. image:: ../diagrams/gclstr_dens_temp_sphere.png

        .. image:: ../diagrams/gclstr_dens_temp_homoeoid.png

        """
        # @@ CONSTRUCT SKELETON @@ #
        # This generates the necessary file structure and background
        # data for the model. All of the structure generation should
        # be the same between coordinate systems.
        profiles = {'gas_density': gas_density, 'temperature': temperature}
        if extra_profiles:
            profiles.update(extra_profiles)

        # build the skeleton for the system and initialize
        # the model object.
        cls.build_skeleton(
            path,
            r_min,
            r_max,
            chunk_shape=chunk_shape,
            num_points=num_points,
            n_phi=n_phi,
            n_theta=n_theta,
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

    # @@ UTILITY FUNCTIONS @@ #
    def _assign_units_and_add_field(self, field_name: str, field_data:unyt.unyt_array, create_field: bool, axes = None):
        """
        Assign correct units to a field and optionally add it to the field container.
        """
        if axes is None:
            axes = ['r']
        units = self.get_default_units(field_name)
        field_data = field_data.to(units)

        if create_field:
            self.FIELDS.add_field(
                field_name,
                axes=axes,
                units=str(units),
                data=field_data.d,
            )
            self.logger.debug("[EXEC] Added field '%s' (units=%s, ndim=%s).", field_name, units,field_data.ndim)

        return field_data

    def _get_radial_spline(self, field_name: str, radius: unyt.unyt_array = None) -> InterpolatedUnivariateSpline:
        """
        Create a radial spline for the specified field.

        Parameters
        ----------
        field_name : str
            The name of the field for which to construct the spline.
        radius : unyt.unyt_array, optional
            The radial coordinates for interpolation.

        Returns
        -------
        InterpolatedUnivariateSpline
            A spline of the specified field over the radial grid.
        """
        # VALIDATE
        if field_name not in self.FIELDS:
            raise ValueError(f"Field '{field_name}' does not exist in the model.")
        if set(self.FIELDS[field_name]._axes) != {'r'}:
            raise ValueError(f"Field '{field_name}' does not have radial axes.")

        if radius is None:
            radius = self._get_radial_coordinates().d
        if hasattr(radius, 'units'):
            radius = radius.d

        # CONSTRUCT the spline.
        field = self.FIELDS[field_name][...].ravel()
        return InterpolatedUnivariateSpline(radius, field.d)

    def _validate_field_dependencies(self, field_name: str, required_fields: dict):
        """
        Validate that all required fields exist for the equation of state computation.
        """
        missing = [f for f in required_fields[field_name] if f not in self.FIELDS]
        if missing:
            raise ValueError(f"Cannot compute {field_name}. Missing fields: {missing}")

    def _get_radial_coordinates(self) -> unyt.unyt_array:
        """
        Retrieve the radial coordinates in the appropriate length units.
        """
        return unyt.unyt_array(self.grid_manager.get_coordinates(axes=['r']).ravel(),
                               self.grid_manager.length_unit)

    def _integrate_hse(self, target_field: str, other_field_spline: InterpolatedUnivariateSpline,
                       density_function: callable, radii: unyt.unyt_array) -> unyt.unyt_array:
        """
        Integrate the hydrostatic equilibrium equation to compute the target field.

        Parameters
        ----------
        target_field : str
            The field being computed ('gravitational_potential' or 'pressure').
        other_field_spline : InterpolatedUnivariateSpline
            Spline representation of the other field (e.g., pressure or gravitational potential).
        density_function : callable
            Function for the gas density profile.
        radii : unyt.unyt_array
            Radial coordinates for integration.

        Returns
        -------
        unyt.unyt_array
            The integrated field.
        """
        # Define the integrand and output units
        if target_field == 'gravitational_potential':
            integrand = lambda r: other_field_spline(r, 1) / density_function(r)
            result_units = self.FIELDS['pressure'].units / self.FIELDS['gas_density'].units
        else:
            integrand = lambda r: other_field_spline(r, 1) * density_function(r)
            result_units = self.FIELDS['gravitational_potential'].units * self.FIELDS['gas_density'].units

        # Integrate the HSE equation
        integrated_field = integrate(integrand, radii.d, x_0=radii.d[-1])

        # Apply boundary conditions
        if target_field == 'gravitational_potential':
            integrated_field -= integrand(radii.d[-1]) * radii.d[-1]
        else:
            boundary_integral = quad(
                lambda r: (other_field_spline(radii.d[-1], 1) * (radii.d[-1] / r) ** 2) / density_function(r),
                radii.d[-1], np.inf
            )[0]
            integrated_field += boundary_integral

        return unyt.unyt_array(integrated_field, result_units)

    def _compute_gradient_hse(self, target_field: str, other_field_spline: InterpolatedUnivariateSpline,
                              density_function: callable, gradient_field_name: str,
                              create_field: bool) -> unyt.unyt_array:
        """
        Compute the gradient field (e.g., gravitational field or pressure gradient).

        Parameters
        ----------
        target_field : str
            The field being computed ('gravitational_potential' or 'pressure').
        other_field_spline : InterpolatedUnivariateSpline
            Spline representation of the other field.
        density_function : callable
            Function for the gas density profile.
        gradient_field_name : str
            The name of the gradient field to compute.
        create_field : bool
            Whether to add the gradient field to the model.

        Returns
        -------
        unyt.unyt_array
            The computed gradient field.
        """
        gradient_axes = self.geometry_handler.get_gradient_dependence(axes=['r'])
        coordinates = self.grid_manager.get_coordinates(axes=gradient_axes)


        # Compute gradient
        gradient = self.geometry_handler.compute_gradient(
            other_field_spline,
            coordinates,
            axes=['r'],
            derivatives=[lambda r: other_field_spline(r, 1)],
            basis='unit',
        )[..., 0]

        # Consistency enforcement
        gradient, density = self.grid_manager.make_fields_consistent(
            [gradient, self.FIELDS['gas_density'][...]],
            [gradient_axes, ['r']]
        )

        # Determine units
        if target_field == 'pressure':
            gradient_field = gradient * density
            gradient_units = (self.FIELDS['gravitational_potential'].units *
                        self.FIELDS['gas_density'].units) / unyt.Unit(self.grid_manager.length_unit)
        else:
            gradient_field = gradient / density
            gradient_units = self.FIELDS['pressure'].units / (self.FIELDS[
                'gas_density'].units * unyt.Unit(self.grid_manager.length_unit))

        gradient_field = unyt.unyt_array(gradient_field, gradient_units)
        return self._assign_units_and_add_field(gradient_field_name, gradient_field, create_field, axes=gradient_axes)

    # @@ CHECKERS @@ #
    # There is only one checker for the galaxy cluster pathways.
    @solver_checker('spherical_dens_temp')
    @solver_checker('spherical_dens_tden')
    @solver_checker('homoeoidal_dens_tden')
    @solver_checker('homoeoidal_dens_temp')
    def check_pathways(self, pathway: str) -> bool:
        """
        Determine if a pathway is valid for a particular set of existing
        profiles and geometry.

        Parameters
        ----------
        pathway: str
            The pathway to validate against.

        Returns
        -------
        bool
            The status of the validation check.
        """
        # Set up the `state` variable to track our status through the checks.
        state = True

        # Check that the coordinate systems are valid.
        # This just relies on checking the coordinate system names.
        cs_name = self.coordinate_system.__class__.__name__
        if pathway.startswith('spherical'):
            state = state and (cs_name == 'SphericalCoordinateSystem')
        if pathway.startswith('homoeidal'):
            state = state and ('Homoeoidal' in cs_name)

        # CHECKING profiles
        # We must have the correct input fields to proceed.
        if pathway.endswith("dens_temp"):
            state = state and ('temperature' in self.profiles)
            state = state and ('gas_density' in self.profiles)
        elif pathway.endswith("dens_tden"):
            state = state and ('total_density' in self.profiles)
            state = state and ('gas_density' in self.profiles)

        return state

    # @@ SOLVER PROCESSES @@ #
    # All of the pipelines are implemented below.
    @solver_process('spherical_dens_temp', step=0)
    @solver_process('spherical_dens_tden', step=0)
    @solver_process('homoeoidal_dens_temp', step=0)
    @solver_process('homoeoidal_dens_tden', step=0)
    def convert_profiles_to_fields(self):
        """
        Convert the provided set of profiles to fields.

        This will ensure that every field in ``self.profiles`` becomes a field
        in the model. Furthermore, if ``stellar_density`` is not provided, then
        a null stellar density is assumed and added as well.
        """
        # Cycle through all of the profiles and add them to the
        # field set for the model.
        for _profile in self.profiles.keys():
            _units = self.get_default_units(_profile)
            self.add_field_from_profile(
                profile_name=_profile,
                chunking=False,
                units=str(_units),
                logging=False
            )
            self.logger.debug("[EXEC] \t\tAdded field `%s` (units=%s) from profile.", _profile, str(_units))

        # Continue by adding the stellar density if it's not already present.
        # This ensures that it can be used in computations elsewhere.
        if 'stellar_density' not in self.FIELDS:
            _units = self.get_default_units('stellar_density')
            self.logger.debug("[EXEC] \t\tAdded field `stellar_density` (units=%s) as null.", str(_units))
            self.FIELDS.add_field(
                'stellar_density',
                axes=['r'],
                units=str(_units),
            )

    @solver_process('homoeoidal_dens_temp',
                    step=6,args=[['gas_density','stellar_density']],kwargs=dict(create_fields=True))
    @solver_process('spherical_dens_temp',
                    step=6,args=[['gas_density','stellar_density','dark_matter_density']],kwargs=dict(create_fields=True))
    @solver_process('homoeoidal_dens_tden',step=2,
                    args=[['gas_density','stellar_density','dark_matter_density','total_density']],kwargs=dict(create_fields=True))
    def integrate_radial_density_field(self, density_field_names: List[str], create_fields: bool = False):
        """
        Integrate radial density profiles to compute corresponding mass profiles.

        This method takes radial density fields (e.g., gas, stellar, dark matter density),
        integrates them over the radial grid, and computes the enclosed mass profiles.

        The resulting mass fields are stored with appropriate units in the grid manager,
        ensuring compatibility with the cluster's physics.

        Parameters
        ----------
        density_field_names : List[str]
            List of names of the density fields to be integrated. Permitted fields include:
            ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density'].
        create_fields : bool, optional
            If True, the integrated mass fields will be added to the grid manager's field container.
            Defaults to False.

        Returns
        -------
        Tuple[unyt.unyt_array, ...]
            A tuple of integrated mass profiles as `unyt.unyt_array` objects.

        Raises
        ------
        ValueError
            If any field name is invalid or not radial in nature.
        KeyError
            If a corresponding mass field name cannot be determined from configuration.

        Notes
        -----
        This method assumes that all density fields provided are defined over radial coordinates (`r` axis).
        """
        # Define valid density fields for validation
        valid_density_fields = ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']
        integrated_mass_fields = []

        # Iterate through the provided density fields
        for density_field_name in density_field_names:
            # Validate the field name
            if density_field_name not in valid_density_fields:
                raise ValueError(f"Field '{density_field_name}' is not a valid density field. "
                                 f"Allowed fields: {valid_density_fields}.")

            # Access the density field from the grid manager
            density_field = self.FIELDS[density_field_name]

            # Validate that the field is radial
            if set(density_field._axes) != {'r'}:
                raise ValueError(f"Field '{density_field_name}' is not defined over the radial ('r') axis.")

            # Retrieve radial coordinates
            radial_coordinates = self.grid_manager.get_coordinates(axes=['r']).ravel()

            # Create an interpolating spline for the density field
            density_spline = InterpolatedUnivariateSpline(radial_coordinates, density_field[...].d)

            # Perform the mass integration in spherical or homoeoidal shells
            enclosed_mass = self.coordinate_system.integrate_in_shells(density_spline, radial_coordinates)

            # Retrieve the corresponding mass field name from configuration
            try:
                mass_field_name = gcluster_params[f'fields.{density_field_name}.mass_field']
            except KeyError:
                raise KeyError(f"Mass field for '{density_field_name}' could not be found in configuration.")

            # Assign units to the computed mass
            mass_units = density_field.units * unyt.Unit(self.grid_manager.length_unit) ** 3
            enclosed_mass = unyt.unyt_array(enclosed_mass, mass_units)

            # Convert units to the default units for the mass field
            target_units = self.get_default_units(mass_field_name)
            enclosed_mass = enclosed_mass.to(target_units)

            # Optionally add the computed mass field to the field container
            mass_field = self._assign_units_and_add_field(mass_field_name, enclosed_mass, create_fields)
            integrated_mass_fields.append(mass_field)

        # Return all integrated mass fields as a tuple
        return tuple(integrated_mass_fields)

    @solver_process('spherical_dens_temp', step=4, args=[['total_density']], kwargs=dict(create_fields=True))
    def compute_density_from_mass_spherical(self, target_fields: List[str], create_fields: bool = False):
        r"""
        Compute the density profile from the corresponding mass profile for specified fields.

        This method calculates the density by differentiating the radial mass profile
        and dividing by the spherical volume element.

        Parameters
        ----------
        target_fields : List[str]
            A list of field names for which to compute the density. Must be one of:
            ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density'].
        create_fields : bool, optional
            If True, the computed density fields are added to the model's field container.
            Defaults to False.

        Returns
        -------
        tuple of unyt.unyt_array
            A tuple containing the computed density fields, one for each field in `target_fields`.

        Raises
        ------
        ValueError
            If a field in `target_fields` is not valid or does not have radial axes.
        KeyError
            If the mass field corresponding to a target field is not defined in the model configuration.

        Notes
        -----
        This method uses the spherical volume element:

            .. math:: \rho(r) = \frac{1}{4 \pi r^2} \frac{dM}{dr}

        where :math:`M` is the cumulative mass profile.
        """
        # Define the list of permitted density fields
        valid_density_fields = ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']

        # Initialize an empty list for the computed density fields
        computed_densities = []

        # Iterate over the requested target fields
        for target_field in target_fields:
            # Validate that the target field is permitted
            if target_field not in valid_density_fields:
                raise ValueError(f"Field '{target_field}' is not a valid density field. "
                                 f"Valid fields are: {valid_density_fields}.")

            # Retrieve the corresponding mass field name from configuration
            try:
                mass_field_name = gcluster_params[f'fields.{target_field}.mass_field']
            except KeyError:
                raise KeyError(f"Configuration missing for field '{target_field}'. "
                               "Could not determine the associated mass field.")

            # Fetch the mass field data
            mass_field = self.FIELDS[mass_field_name]
            if set(mass_field._axes) != {'r'}:
                raise ValueError(f"Field '{mass_field_name}' does not have radial axes. "
                                 "Only radial mass fields are supported.")

            # Retrieve the radial coordinates
            radii = self._get_radial_coordinates()

            # Create a spline for the radial mass profile
            mass_spline = InterpolatedUnivariateSpline(radii, mass_field[...].d)

            # Compute the density by differentiating the mass profile
            density_values = mass_spline.derivative(n=1)(radii.d) / (4 * np.pi * radii ** 2)
            density_units = mass_field.units / radii.units ** 3
            density_field = unyt.unyt_array(density_values, density_units)

            # Assign units and optionally add the density field to the container
            computed_field = self._assign_units_and_add_field(target_field, density_field, create_fields, axes=['r'])
            computed_densities.append(computed_field)

        return tuple(computed_densities)

    @solver_process('homoeoidal_dens_temp', step=7, args=['dark_matter_mass'], kwargs=dict(create_field=True, field_type='mass'))
    @solver_process('spherical_dens_temp', step=5, args=['dark_matter_density'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_temp', step=5, args=['dark_matter_density'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_tden',step=1,args=['dark_matter_density'], kwargs=dict(create_field=True))
    def compute_missing_mass_field(self, target_field: str, create_field: bool = False, field_type: str = 'density'):
        """
        Compute a missing field (mass or density) by summing existing components
        and subtracting from the total field if necessary.

        This method generalizes the computation of a missing mass or density field
        based on the provided field type.

        Parameters
        ----------
        target_field : str
            The name of the field to compute. For density fields:
            ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density'].
            For mass fields:
            ['total_mass', 'gas_mass', 'stellar_mass', 'dark_matter_mass'].
        create_field : bool, optional
            If True, the computed field is added to the model's field container. Defaults to False.
        field_type : str, optional
            The type of field to compute. Must be either 'density' or 'mass'. Defaults to 'density'.

        Returns
        -------
        unyt.unyt_array
            The computed field with appropriate units.

        Raises
        ------
        ValueError
            If `target_field` is invalid or `field_type` is not 'density' or 'mass'.

        Notes
        -----
        - If `target_field` is 'total_mass' or 'total_density', the method sums up all components.
        - For other fields (e.g., 'dark_matter_mass' or 'dark_matter_density'), the missing
          field is computed by subtracting the sum of components from the total field.
        """
        # VALIDATE that the field is a valid field and that the field type is also valid.
        if field_type == 'density':
            valid_fields = ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']
            _universal_unit = 'Msun/kpc**3'
        elif field_type == 'mass':
            valid_fields = ['total_mass', 'gas_mass', 'stellar_mass', 'dark_matter_mass']
            _universal_unit = 'Msun'
        else:
            raise ValueError("Invalid `field_type`. Must be 'density' or 'mass'.")

        if target_field not in valid_fields:
            raise ValueError(f"'{target_field}' is not a valid {field_type} field. "
                             f"Valid fields are: {valid_fields}.")

        # RETRIEVE all of the existing fields of this type using the list of
        # 'valid_fields' that we already have defined.
        existing_fields = {
            name: self.FIELDS[name][...].to_value(_universal_unit) for name in valid_fields if name != target_field
        }

        # ENFORCE grid consistency conditions. This ensures that we can perform the
        # necessary operations even when grids don't have the same number of axes.
        existing_field_axes = [self.FIELDS[name]._axes for name in existing_fields.keys()]
        consistent_fields = self.grid_manager.make_fields_consistent(
            list(existing_fields.values()), existing_field_axes
        )

        # determine the axes as a set.
        axes_set = self.coordinate_system.ensure_axis_order(set().union(*existing_field_axes))
        axes_mask = np.array([cax in axes_set for cax in self.coordinate_system.AXES], dtype=bool)
        new_shape = self.grid_manager.GRID_SHAPE[axes_mask]
        existing_fields = {name: np.broadcast_to(consistent_fields[i],new_shape) for i, name in enumerate(existing_fields)}

        # COMPUTE the sum of the existing fields. This may then be either added or removed from
        # the total to compute the output value.

        field_sum = np.sum(np.stack(list(existing_fields.values()), axis=0), axis=0)
        if target_field == valid_fields[0]:  # 'total_mass' or 'total_density'
            computed_field = unyt.unyt_array(field_sum, units=_universal_unit)
        else:
            total_field = self.FIELDS[valid_fields[0]][...].to_value(_universal_unit)  # 'total_mass' or 'total_density'
            computed_field = unyt.unyt_array(total_field - field_sum, units=_universal_unit)

        # Assign correct units and optionally add the field
        result_field = self._assign_units_and_add_field(
            target_field, computed_field, create_field, axes=axes_set
        )
        return result_field

    @solver_process('spherical_dens_temp', step=2,args=['gravitational_potential'],kwargs=dict(create_field=True,
                                                                                                add_gradient_field=True))
    @solver_process('homoeoidal_dens_temp', step=2,args=['gravitational_potential'],kwargs=dict(create_field=True,
                                                                                                add_gradient_field=True))
    def solve_hse(self, field_name: str, create_field: bool = False, add_gradient_field: bool = False):
        """
        Solve for hydrostatic equilibrium (HSE) to compute either the gravitational potential
        or the pressure field, optionally calculating the gradient field.

        Parameters
        ----------
        field_name : str
            The field to compute. Must be one of ['gravitational_potential', 'pressure'].
        create_field : bool, optional
            If True, the computed field is added to the model's field container. Defaults to False.
        add_gradient_field : bool, optional
            If True, also compute and add the gradient of the corresponding field. Defaults to False.

        Returns
        -------
        unyt.unyt_array or tuple of unyt.unyt_array
            The computed HSE field. If `add_gradient_field` is True, returns a tuple containing:
            - The computed HSE field.
            - The computed gradient field.

        Raises
        ------
        ValueError
            If `target_field` is not 'gravitational_potential' or 'pressure'.

        Notes
        -----
        The method integrates the HSE equation:
            - For `gravitational_potential`, the pressure gradient is used.
            - For `pressure`, the density and gravitational field are used.

        It optionally computes the gradient of the field, such as the gravitational field
        (gradient of gravitational potential) or the pressure gradient.
        """
        # Validate target field
        valid_fields = ['gravitational_potential', 'pressure']
        if field_name not in valid_fields:
            raise ValueError(f"Invalid target_field '{field_name}'. Must be one of {valid_fields}.")

        # Determine corresponding field for integration
        other_field = 'pressure' if field_name == 'gravitational_potential' else 'gravitational_potential'
        gradient_field_name = 'gravitational_field' if field_name == 'gravitational_potential' else 'pressure_gradient'

        # Fetch radial coordinates and field splines
        radii = self._get_radial_coordinates()
        other_field_spline = self._get_radial_spline(other_field)
        density_function = self.profiles['gas_density']

        # Integrate HSE field
        hse_field = self._integrate_hse(field_name, other_field_spline, density_function, radii)

        # Add the HSE field to the field container if requested
        hse_field = self._assign_units_and_add_field(field_name, hse_field, create_field, axes=['r'])

        # Compute gradient field if requested
        if add_gradient_field:
            gradient_field = self._compute_gradient_hse(
                field_name, other_field_spline, density_function, gradient_field_name, create_field
            )
            return hse_field, gradient_field

        return hse_field


    @solver_process('spherical_dens_temp', step=1, args=['pressure'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_temp', step=1, args=['pressure'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_tden', step=5, args=['temperature'], kwargs=dict(create_field=True))
    def solve_eos(self, field_name: str, create_field: bool = False):
        """
        Solve for a thermodynamic field using the equation of state.

        Parameters
        ----------
        field_name : str
            The name of the field to solve for: 'temperature', 'pressure', or 'gas_density'.
        create_field : bool, optional
            If True, the computed field is added to the FIELDS container.

        Returns
        -------
        unyt.unyt_array
            The computed thermodynamic field.
        """
        # Validate input field and required dependencies
        required_fields = {'temperature': ['pressure', 'gas_density'],
                           'pressure': ['temperature', 'gas_density'],
                           'gas_density': ['pressure', 'temperature']}
        self._validate_field_dependencies(field_name, required_fields)

        # Retrieve required fields
        pressure = self.FIELDS.get('pressure', None)
        density = self.FIELDS.get('gas_density', None)
        temperature = self.FIELDS.get('temperature', None)
        scale_factor = m_p * mu  # Universal scale factor

        # Compute the desired field
        if field_name == 'temperature':
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(pressure._axes,density._axes)]
            pressure, density = make_grid_fields_broadcastable([pressure[...],density[...]],
                                                               [pressure._axes,density._axes],
                                                               self.coordinate_system)
            field = (pressure * scale_factor) / density
        elif field_name == 'pressure':
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(temperature._axes,density._axes)]
            temperature, density = make_grid_fields_broadcastable([temperature[...],density[...]],
                                                               [temperature._axes,density._axes],
                                                               self.coordinate_system)
            field = (temperature * density) / scale_factor
        elif field_name == 'gas_density':
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(pressure._axes,temperature._axes)]
            pressure, temperature = make_grid_fields_broadcastable([pressure[...],temperature[...]],
                                                               [pressure._axes,temperature._axes],
                                                               self.coordinate_system)
            field = (pressure[...] * scale_factor) / temperature[...]
        else:
            raise ValueError(f"Invalid field name '{field_name}' provided.")

        # Set units and create the field if required
        field = self._assign_units_and_add_field(field_name, field, create_field, axes)
        return field

    @solver_process('homoeoidal_dens_temp', step=3)
    def compute_tdens_from_potential(self):
        r"""
        Compute the total density field from the gravitational potential using the Laplacian.

        This method computes the total density by applying the Laplacian operator to the
        gravitational potential field. The resulting Laplacian is scaled to obtain the density
        via the Poisson equation:

            .. math:: \nabla^2 \Phi = 4 \pi G \rho

        where :math:`\Phi` is the gravitational potential, :math:`\rho` is the total density,
        and :math:`G` is the gravitational constant.

        Returns
        -------
        None
            The computed total density field is added to the model's field container under
            the name `total_density`.

        Notes
        -----
        This method assumes radial symmetry and computes derivatives (first and second) of
        the gravitational potential using splines and a geometry handler to evaluate the
        Laplacian.
        """
        # Retrieve the radius and the potential spline. Compute the necessary derivatives
        # so that they can be fed to the Laplacian.
        radius = self._get_radial_coordinates()
        potential_spline = self._get_radial_spline('gravitational_potential')

        first_derivative_spline = InterpolatedUnivariateSpline(radius.d, potential_spline(radius.d, 1))
        second_derivative = lambda r: first_derivative_spline(r, 1)

        # Compute the laplacian
        laplacian_axes = self.geometry_handler.get_laplacian_dependence(axes=['r'])
        coordinates = self.grid_manager.get_coordinates(axes=laplacian_axes)
        laplacian = self.geometry_handler.compute_laplacian(
            potential_spline,
            coordinates,
            axes=['r'],
            first_derivatives=[first_derivative_spline],
            second_derivatives=[second_derivative],
            edge_order=2
        )

        # Convert Laplacian to unyt array with proper units
        laplacian_units = self.FIELDS['gravitational_potential'].units / radius.units ** 2
        laplacian = unyt.unyt_array(laplacian, laplacian_units)

        # SCompute total density using the Poisson equation: rho = laplacian / (4 * pi * G)
        total_density = laplacian / (4 * np.pi * G)
        self._assign_units_and_add_field('total_density', total_density, create_field=True, axes=laplacian_axes)

    @solver_process('homoeoidal_dens_temp', step=4)
    def compute_total_mass_gauss_nonspherical(self):
        r"""
        Compute the total mass in a non-spherical coordinate system using Gauss's theorem.

        This method calculates the total mass profile based on the gradient of the gravitational
        potential in non-spherical geometries (e.g., oblate or prolate homoeoidal systems).
        The mass is determined by:

            .. math:: M(r) = \frac{f \cdot r^2 \cdot \partial_r \Phi}{4 \pi G}

        where :math:`\Phi` is the gravitational potential, :math:`f` is the geometry-specific
        flux factor, and :math:`G` is the gravitational constant.

        Returns
        -------
        None
            The computed total mass field is added to the model's field container under
            the name `total_mass`.

        Notes
        -----
        The flux factor `f` is specific to the geometry and is retrieved from the
        coordinate system's attributes.
        """
        # Retrieve the necessary data and compute the flux parameter. Get the potential derivative.
        radius = self._get_radial_coordinates()
        potential_spline = self._get_radial_spline('gravitational_potential')
        flux_factor = self.coordinate_system.flux_parameter
        d_potential_dr = potential_spline(radius.d, 1)  # First derivative wrt radius

        # Compute the numerator for the total mass equation
        numerator = flux_factor * d_potential_dr * radius.d ** 2
        numerator_units = radius.units * self.FIELDS['gravitational_potential'].units
        numerator = unyt.unyt_array(numerator, numerator_units)

        # Compute total mass using Gauss's theorem
        total_mass = numerator / (4 * np.pi * G)
        self._assign_units_and_add_field('total_mass', total_mass, create_field=True, axes=['r'])

    @solver_process('spherical_dens_temp', step=3)
    def compute_total_mass_gauss_spherical(self):
        r"""
        Compute the total mass in a spherical coordinate system using Gauss's theorem.

        This method calculates the total mass profile based on the gravitational field
        in spherical geometry. The mass is determined by:

            .. math:: M(r) = - \frac{r^2 \cdot g(r)}{G}

        where :math:`g(r)` is the radial gravitational field, :math:`G` is the gravitational
        constant, and :math:`r` is the radial coordinate.

        Returns
        -------
        None
            The computed total mass field is added to the model's field container under
            the name `total_mass`.

        Notes
        -----
        This method assumes spherical symmetry, where the gravitational field
        directly relates to the mass enclosed within a radius `r`.
        """
        radius = self._get_radial_coordinates()
        gravitational_field = self.FIELDS['gravitational_field'][...]
        total_mass = (-radius ** 2 * gravitational_field) / G
        self._assign_units_and_add_field('total_mass', total_mass, create_field=True, axes=['r'])

    @solver_process('homoeoidal_dens_tden',step=3)
    def solve_poisson_problem_homoeoidal(self):
        # Pull out the coordinates and the density profile from the Model instance.
        coordinates = self.grid_manager.get_coordinates()
        density_profile = self.profiles['total_density']
        _dunits,_lunits = unyt.Unit(density_profile.units),unyt.Unit(self.grid_manager.length_unit)

        # Compute the gravitational potential. This passes down to the poisson solver at the
        # lower level of the coordinate system object.
        # TODO: this should be unifiable using symmetry; however, the Handler object cannot tell
        # that it's a radial handler. This needs to be managed...
        gravitational_potential = self.coordinate_system.solve_radial_poisson_problem(
            density_profile,
            coordinates,
        )
        gravitational_potential = G*unyt.unyt_array(gravitational_potential[...,0], _dunits*_lunits**2) # cut of phi dependence.

        # Assign the data to a the gravitational potential field and proceed.
        self._assign_units_and_add_field('gravitational_potential', gravitational_potential, True, ['r','theta'])

    @solver_process('homoeoidal_dens_tden',step=4)
    def solve_hse_asymmetric(self):
        # Pull the coordinates and the units necessary. Because we are integrating radially, only
        # the radii matter, not the angular coordinates.
        radius = self._get_radial_coordinates()
        gas_density_profile = self.profiles['gas_density']
        gravitational_potential = self.FIELDS['gravitational_potential'][...]

        # Build the splines that are necessary for our integration scheme.
        # - gas_density_spline: rho -- we only need this for the derivative.
        # - middle_integrand_spline: d(rho)/dr * phi
        gas_density_spline = InterpolatedUnivariateSpline(radius.d, gas_density_profile(radius.d))

        # construct the middle_integrand_spline
        deriv_gas_dens = gas_density_spline(radius.d,1)[...,np.newaxis] #-> reshaped to ensure broadcastability
        middle_integrand_spline = make_interp_spline(radius.d, deriv_gas_dens*gravitational_potential.d)

        # Compute the integrals for the pressure.
        # 1. -rho(r)*phi(r)
        # 2. - int_r^r_max d(rho)/dr * phi(r) dr
        # 3. - int_r_max^inf d(rho)/dr * phi(r) dr.
        extended_potential = lambda _r: gravitational_potential[-1,:].d * (_r/radius.d[-1])**(-1)
        middle_integrand = lambda _r: middle_integrand_spline(_r)
        outer_integrand = lambda _r: gas_density_spline(_r,1)*extended_potential(_r)

        # compute the relevant integrals
        inner_integral = gas_density_spline(radius.d)[...,np.newaxis] * gravitational_potential.d
        middle_integral = integrate_vectorized(middle_integrand, radius.d, x_0=radius.d[-1])
        outer_integral = quad_vec(outer_integrand, radius.d[-1], np.inf)[0]

        # Set the pressure and coerce the units
        pressure = -inner_integral - middle_integral # TODO: the boundary needs to be dealt with.
        base_units = self.FIELDS['gravitational_potential'].units*self.FIELDS['gas_density'].units
        pressure = unyt.unyt_array(pressure,base_units)

        # Pass to the unit setter and field adder.
        self._assign_units_and_add_field('pressure', pressure, True, ['r','theta'])


if __name__ == '__main__':
    from pisces.profiles import NFWDensityProfile, IsothermalTemperatureProfile
    from pisces.geometry import SphericalCoordinateSystem, OblateHomoeoidalCoordinateSystem

    print(ClusterModel.list_pipelines())
    d = NFWDensityProfile(rho_0=1e5, r_s=100)
    td = NFWDensityProfile(rho_0=1e6, r_s=150)
    t = IsothermalTemperatureProfile(T_0=5)

    cs = SphericalCoordinateSystem()

    cs_ho = OblateHomoeoidalCoordinateSystem(ecc=0.9)

    model_hom = ClusterModel.from_dens_and_tden('test_hom.hdf5', 1e-1, 1e4, d, td,coordinate_system=cs_ho,n_theta=50,overwrite=True)
    #model = ClusterModel.from_dens_and_temp('test.hdf5', 1e-1, 1e4, d, t, coordinate_system=cs, overwrite=True)
    import matplotlib.pyplot as plt
    r = model_hom._get_radial_coordinates().d
    plt.semilogx(r,model_hom.FIELDS['temperature'][:,:])
    plt.show()

