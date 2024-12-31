from pathlib import Path
from typing import Union, List, Optional, Dict, TYPE_CHECKING

import numpy as np
import unyt
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.integrate import quad_vec
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import make_interp_spline

from pisces.models.utilities import ModelConfigurationDescriptor
from pisces.utilities.array_utils import make_grid_fields_broadcastable
from pisces.geometry import CoordinateSystem
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
from pisces.io import HDF5_File_Handle
from pisces.models.base import Model
from pisces.models.galaxy_clusters.grids import ClusterGridManager
from pisces.models.galaxy_clusters.utils import gcluster_params
from pisces.models.solver import solver_checker, solver_process, serial_solver_checkers, serial_solver_processes
from pisces.utilities.math_utils.numeric import integrate_vectorized
from pisces.utilities.physics import m_p, mu, G
from pisces.utilities.math_utils.numeric import integrate

if TYPE_CHECKING:
    from pisces.profiles.base import Profile
    from pisces.geometry.base import RadialCoordinateSystem


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
    GRID_MANAGER_TYPE = ClusterGridManager
    INIT_FREE_AXES = ['r']
    config = ModelConfigurationDescriptor(filename='galaxy_clusters.yaml')

    @property
    def coordinate_system(self) -> 'RadialCoordinateSystem':
        # noinspection PyTypeChecker
        # skip the type checking because we have enforcements on the
        # coordinate system which are dynamic.
        return super().coordinate_system

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
                       length_unit: str = None,
                       scale: Union[List[str], str] = None,
                       profiles: Optional[Dict[str, 'Profile']] = None,
                       coordinate_system: Optional[SphericalCoordinateSystem] = None) -> 'HDF5_File_Handle':
        r"""
        Construct a "skeleton" for the :py:class:`ClusterModel` class.

        The skeleton is the base structure necessary to load an HDF5 file as this object. This includes the following basic
        structures:

        - The :py:class:`~pisces.models.grids.base.ModelGridManager` instance, which controls the grids and data storage.
        - The :py:class:`~pisces.profiles.collections.HDF5ProfileRegistry` instance, which acts as a container for the user
          provided profiles.

        .. important::

            In general use, the :py:meth:`build_skeleton` method should only rarely be called. When generating a model subclass
            for a particular physical system, the user should generate the model by calling a "generator" method of the class. The
            generator method then performs some basic setup tasks before passing a more constrained set of arguments to the
            :py:meth:`build_skeleton` method.

        Parameters
        ----------
        path : str or Path
            The path at which to build the skeleton for the model. If a file already exists at ``path``, then an error
            is raised unless ``overwrite=True``. If ``overwrite=True``, then the original file is deleted and the new skeleton
            will take its place.
        r_min: float
            The minimum radius of the galaxy cluster base grid in units of ``length_unit``. This value must be larger than zero
            to prevent common issues with division by zero, but otherwise may be as small as is desired.
        r_max: float
            The maximum radius of the galaxy cluster in units of ``length_unit``. Generally, maximum radii on the order of 5000 or
            more kpc are suggested for physical reliability.
        num_points: int, optional
            The number of radial points in the galaxy cluster's grid. By default, ``num_points = 1000``.
        n_phi: int, optional
            The number of grid points to place in the :math:`\phi` axis. This attribute is **only required** when
            using a triaxial coordinate system; otherwise, it is redundant and is unused.
        n_theta: int, optional
            The number of grid points to place in the :math:`\theta` axis. This attribute is **only required** when
            using a non-spherical coordinate system; otherwise, it is redundant and is unused.
        chunk_shape : ArrayLike, optional
            The shape of each chunk used for subdividing the grid, allowing chunk-based
            operations or partial in-memory loading.

            If not provided, it defaults to
            ``grid_shape``, meaning the entire grid is treated as a single chunk. If specified,
            each element must divide the corresponding element in ``grid_shape`` without
            remainder.

            .. important::

                The choice to perform operations in chunks (or not to) is made by the developer of
                the relevant model. Generally, if it isn't necessary to perform operations in chunks, it's avoided.
                As such, it's generally advisable to leave this argument unchanged unless you have a clear reason
                to set it.
        overwrite : bool, optional
            If True, overwrite any existing file at the specified path. If False and the
            file exists, an exception will be raised. Defaults to False.
        length_unit : str, optional
            The physical length unit for interpreting grid coordinates, for example `"kpc"`
            or `"cm"`. Defaults to the :py:attr:`DEFAULT_LENGTH_UNIT` of this model class's :py:attr:`GRID_MANAGER_TYPE`.

        scale : Union[List[str], str], optional
            The scaling mode for each axis, determining whether cells are spaced linearly or
            logarithmically. Each entry can be `"linear"` or `"log"`. If a single string is given,
            it is applied to all axes. Defaults to the :py:attr:`DEFAULT_SCALE` of this model class's :py:attr:`GRID_MANAGER_TYPE`.

        profiles : dict[str, :py:class:`~pisces.profiles.base.Profile`], optional
            A dictionary containing profiles to register as part of the model. Keys in the dictionary should correspond
            to the name of the physical quantity being described by the corresponding profile.

            The profiles provided are saved to the skeleton at ``path``. They are then accessible via the :py:attr:`profiles` attribute
            of the created :py:class:`Model` instance.

            .. tip::

                At its core, the :py:class:`Model` has no expectations on the profiles that are provided. They are all
                registered directly using the name / value provided in ``profiles``. In many subclasses, the accessible
                solution pathways may be dictated (partially or in whole) by what profiles the user registered upon
                initializing the class. Generally, these are accompanied with "generator methods", which handle the naming
                and registration for the user.

        coordinate_system : :py:class:`~pisces.geometry.base.CoordinateSystem`, optional
            The coordinate system that defines the dimensionality and axes of the grid. If a coordinate system is not
            provided, then the default coordinate system (:py:attr:`DEFAULT_COORDINATE_SYSTEM`) will be used. If there is
            no default coordinate system for this class, then an error is raised.

        Returns
        -------
        HDF5_File_Handle
            The HDF5 file handle.

        Raises
        ------
        ValueError
            If required parameters are missing or validation fails.
        """
        # Initialize the coordinate system including the default implementation if the
        # coordinate system isn't actually provided. NOTE: we don't defer to super() here because
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

        # fix the scale to meet defaults.
        if scale is None:
            scale = cls.GRID_MANAGER_TYPE.DEFAULT_SCALE
        scale = cls.GRID_MANAGER_TYPE.correct_scale(scale)
        bbox = cls.GRID_MANAGER_TYPE.correct_bbox(r_min, r_max)

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
            raise NotImplementedError(
                f"The coordinate system {obj.coordinate_system.__class__.__name__} is an accepted coordinate"
                f" system for {cls.__name__}, but there is no density / temperature pipeline implemented.")

        return obj

    # @@ UTILITY FUNCTIONS @@ #
    # These utility functions are used throughout the model generation process for various things
    # and are not sufficiently general to be worth implementing elsewhere.
    def _assign_units_and_add_field(
            self,
            field_name: str,
            field_data: unyt.unyt_array,
            create_field: bool,
            axes: Optional[List[str]] = None
    ) -> unyt.unyt_array:
        """
        Assigns the appropriate units to a field and optionally adds it to the model's field container.

        This utility method ensures that a given field has the correct physical units before being
        integrated into the model's field container. It facilitates consistency and correctness
        in the model's physical quantities by enforcing unit assignments and managing field
        registrations.

        Parameters
        ----------
        field_name : str
            The name of the field to process. This name is used both for logging purposes and
            when adding the field to the model's field container.

        field_data : unyt.unyt_array
            The raw data of the field as a `unyt.unyt_array`. This array contains both the
            numerical values and their associated units.

        create_field : bool
            A flag indicating whether the processed field should be added to the model's
            field container (`self.FIELDS`). If `True`, the field is registered; if `False`,
            the field is only processed for unit consistency.

        axes : Optional[List[str]], default=None
            A list of axis names that the field depends on (e.g., `['r']` for radial dependence).
            If not provided, it defaults to `['r']`, assuming radial dependence.

        Returns
        -------
        unyt.unyt_array
            The processed field data with the correct units assigned.

        Raises
        ------
        ValueError
            If the specified `field_name` does not have a corresponding default unit defined
            within the model. This ensures that all fields have predefined unit expectations.
        """

        if axes is None:
            axes = ['r']

            # Retrieve the default units for the specified field
        try:
            units = self.get_default_units(field_name)
        except KeyError as e:
            raise ValueError(f"No default units defined for field '{field_name}'.") from e

            # Convert the field data to the default units
        field_data = field_data.to(units)

        if create_field:
            # Add the field to the model's field container
            self.FIELDS.add_field(
                field_name,
                axes=axes,
                units=str(units),
                data=field_data.d,
            )
            self.logger.debug(
                "[EXEC] Added field '%s' (units=%s, ndim=%s).",
                field_name,
                units,
                field_data.ndim
            )

        return field_data

    def _get_radial_spline(
            self,
            field_name: str,
            radius: Optional[unyt.unyt_array] = None
    ) -> InterpolatedUnivariateSpline:
        """
        Generate a radial spline for the specified field.

        Parameters
        ----------
        field_name : str
            Name of the field to spline.
        radius : unyt.unyt_array, optional
            Radial coordinates for interpolation. Defaults to the model's radial coordinates.

        Returns
        -------
        InterpolatedUnivariateSpline
            Spline of the field over the radial grid.

        Raises
        ------
        ValueError
            If the field does not exist or is not defined over the radial axis.
        """
        if field_name not in self.FIELDS:
            raise ValueError(f"Field '{field_name}' does not exist in the model.")
        if set(self.FIELDS[field_name].AXES) != {'r'}:
            raise ValueError(f"Field '{field_name}' is not defined over the radial axis.")

        if radius is None:
            radius = self._get_radial_coordinates().d
        else:
            radius = radius.d if hasattr(radius, 'units') else radius

        field_data = self.FIELDS[field_name][...].ravel()
        return InterpolatedUnivariateSpline(radius, field_data.d)

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

    def _integrate_hse_radial(self,
                              target_field: str,
                              other_field_spline: InterpolatedUnivariateSpline,
                              density_function: callable,
                              radii: unyt.unyt_array) -> unyt.unyt_array:
        """
        Integrate the hydrostatic equilibrium equation to compute the ``target field``.

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
        # Define the integrand and output units. This requires knowledge of the target field;
        # we can then look everything up to use the formula correctly.
        # The integrand is then the element that will be integrated to obtain the target field.
        if target_field == 'gravitational_potential':
            # The other field is the pressure, dphi = dP/rho
            integrand = lambda _r: other_field_spline(_r, 1) / density_function(_r)
            result_units = self.FIELDS['pressure'].units / self.FIELDS['gas_density'].units
        elif target_field == 'pressure':
            # The other field is potential, dP = dphi*rho
            integrand = lambda _r: other_field_spline(_r, 1) * density_function(_r)
            result_units = self.FIELDS['gravitational_potential'].units * self.FIELDS['gas_density'].units
        else:
            raise ValueError(f"Target field '{target_field}' not recognized.")

        # Perform the integration step using the ``integrate`` method to go from the
        # radii out to the maximum radius.
        integrated_field = integrate(integrand, radii.d, x_0=radii.d[-1])

        # Apply boundary conditions:
        # For the gravitational potential, we assume dP/rho ~ -dphi ~ -r^-2 at large radii. Under assumption
        #   of continuity, this leads to the closed form applied.
        # For the pressure, we assume dphi ~ r^-2, which leads to a well constrained quadrature rule at large
        #   radii. This takes advantage of the fact that we know the density function with arb. precision.
        if target_field == 'gravitational_potential':
            integrated_field -= integrand(radii.d[-1]) * radii.d[-1]
        elif target_field == 'pressure':
            boundary_integral = quad(
                lambda _r: (other_field_spline(radii.d[-1], 1) * (radii.d[-1] / _r) ** 2) / density_function(_r),
                radii.d[-1], np.inf
            )[0]
            integrated_field += boundary_integral
        else:
            raise ValueError(f"Target field '{target_field}' not recognized.")

        # Return the output as an array with the result units.
        return unyt.unyt_array(integrated_field, result_units)

    def _compute_gradient_hse(self, target_field: str,
                              other_field_spline: InterpolatedUnivariateSpline,
                              gradient_field_name: str,
                              create_field: bool) -> unyt.unyt_array:
        """
        Compute the gradient field (e.g., gravitational field or pressure gradient).
        This mirrors the _integrate_hse_radial in many respects; however, it is compatible
        with higher dimensional gradient computations (non-radial geometries).

        Parameters
        ----------
        target_field : str
            The field being computed ('gravitational_potential' or 'pressure').
        other_field_spline : InterpolatedUnivariateSpline
            Spline representation of the other field.
        gradient_field_name : str
            The name of the gradient field to compute.
        create_field : bool
            Whether to add the gradient field to the model.

        Returns
        -------
        unyt.unyt_array
            The computed gradient field.
        """
        # Utilize the geometry handler (with the default fixed axes = 'r') to determine how the
        # gradient will manifest. Then fetch the necessary coordinate grid for the operation.
        gradient_axes = self.geometry_handler.get_gradient_dependence(axes=['r'])
        coordinates = self.grid_manager.get_coordinates(axes=gradient_axes)

        # Compute the gradient of the spline that was provided as the other field spline. This
        # will be either the gravitational potential spline or the pressure spline.
        # NOTE that we take the zero component because the returned grid is (..., 1) in shape -> (...,) in final
        # shape.
        gradient = self.geometry_handler.compute_gradient(
            other_field_spline,
            coordinates,
            axes=['r'],
            derivatives=[lambda _r: other_field_spline(_r, 1)],
            basis='unit',
        )[..., 0]

        # Ensure that the fields are broadcastable to one another. This is
        # required if the the gradient breaks symmetry and has a larger dimension that
        # the density field does.
        gradient, density = self.grid_manager.make_fields_consistent(
            [gradient, self.FIELDS['gas_density'][...]],
            [gradient_axes, ['r']]
        )

        # Determine units -> This is a simple application of the
        # HSE equation to determine dimensionality. Use the length unit to manage
        # the unit manipulations from differential operations.
        if target_field == 'pressure':
            gradient_field = gradient * density
            gradient_units = (self.FIELDS['gravitational_potential'].units *
                              self.FIELDS['gas_density'].units) / unyt.Unit(self.grid_manager.length_unit)
        elif target_field == 'gravitational_potential':
            gradient_field = gradient / density
            gradient_units = self.FIELDS['pressure'].units / (self.FIELDS[
                                                                  'gas_density'].units * unyt.Unit(
                self.grid_manager.length_unit))
        else:
            raise ValueError("Unknown target field.")

        # Enforce units and register the new field.
        gradient_field = unyt.unyt_array(gradient_field, gradient_units)
        return self._assign_units_and_add_field(gradient_field_name, gradient_field, create_field, axes=gradient_axes)

    # @@ CHECKERS @@ #
    # There is only one checker for the galaxy cluster pathways.
    @serial_solver_checkers([
        'spherical_dens_temp',
        'spherical_dens_tden',
        'homoeoidal_dens_tden',
        'homoeoidal_dens_temp'
    ])
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
    @serial_solver_processes([
        ('spherical_dens_temp', 0,[],{}),
        ('spherical_dens_tden', 0,[],{}),
        ('homoeoidal_dens_temp',0,[],{}),
        ('homoeoidal_dens_tden',0,[],{}),
    ])
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
                _profile,
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

    @serial_solver_processes([
        ('homoeoidal_dens_temp', 6, [['gas_density', 'stellar_density']], dict(create_fields=True)),
        ('spherical_dens_temp' , 6, [['gas_density', 'stellar_density','dark_matter_density']], dict(create_fields=True)),
        ('homoeoidal_dens_tden', 2, [['gas_density', 'stellar_density','dark_matter_density','total_density']], dict(create_fields=True)),
        ('spherical_dens_tden', 2, [['gas_density', 'stellar_density', 'dark_matter_density', 'total_density']],dict(create_fields=True)),
    ])
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
        # Define valid density fields for validation. Fields not in the valid set are checked for errors.
        valid_density_fields = ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']

        # Cycle through each of the provided density fields and perform the integration.
        # The final product is appended to the `integrated_mass_fields` to eventually return
        # after cycle finishes.
        integrated_mass_fields = []
        for density_field_name in density_field_names:
            # Retrieve the field from the FIELD container.
            density_field = self.FIELDS[density_field_name]

            # Validate the pulled density field. We cannot handle non-radial fields, so
            # we need to check for that. Additionally, it needs to be a valid density field.
            if density_field_name not in valid_density_fields:
                raise ValueError(f"Field '{density_field_name}' is not a valid density field. "
                                 f"Allowed fields: {valid_density_fields}.")
            if set(density_field.AXES) != {'r'}:
                raise ValueError(f"Field '{density_field_name}' is not defined over the radial ('r') axis.")

            # Look up the corresponding mass field in the configuration file to
            # ensure that we know the name and the units.
            # Retrieve the corresponding mass field name from configuration
            try:
                mass_field_name = self.config[f'fields.{density_field_name}.mass_field']
            except KeyError:
                raise KeyError(f"Mass field for '{density_field_name}' could not be found in configuration.")

            # Construct the coordinates, build the interpolated spline and then
            # perform the shell integration routine. This should be adaptable
            # for any radial coordinate system due to the implementation of
            # integrate_in_shells.
            radial_coordinates = self.grid_manager.get_coordinates(axes=['r']).ravel()
            density_spline = InterpolatedUnivariateSpline(radial_coordinates, density_field[...].d)
            enclosed_mass = self.coordinate_system.integrate_in_shells(density_spline, radial_coordinates)


            # Manage the units. We compute `mass_units` which is the natural unit of the
            # computation and then covert to a target unit.
            mass_units = density_field.units * unyt.Unit(self.grid_manager.length_unit) ** 3
            enclosed_mass = unyt.unyt_array(enclosed_mass, mass_units)
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
            if set(mass_field.AXES) != {'r'}:
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

    @serial_solver_processes([
        ('homoeoidal_dens_temp', 5, ['dark_matter_density'], dict(create_field=True)),
        ('homoeoidal_dens_temp', 7, ['dark_matter_mass']   ,dict(create_field=True, field_type='mass')),
        ('spherical_dens_temp' , 5, ['dark_matter_density'], dict(create_field=True)),
        ('homoeoidal_dens_tden', 1, ['dark_matter_density'], dict(create_field=True)),
        ('spherical_dens_tden',  1, ['dark_matter_density'], dict(create_field=True)),
    ])
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
        existing_field_axes = [self.FIELDS[name].AXES for name in existing_fields.keys()]
        consistent_fields = self.grid_manager.make_fields_consistent(
            list(existing_fields.values()), existing_field_axes
        )

        # determine the axes as a set.
        axes_set = self.coordinate_system.ensure_axis_order(set().union(*existing_field_axes))
        axes_mask = np.array([cax in axes_set for cax in self.coordinate_system.AXES], dtype=bool)
        new_shape = self.grid_manager.GRID_SHAPE[axes_mask]
        existing_fields = {name: np.broadcast_to(consistent_fields[i], new_shape) for i, name in
                           enumerate(existing_fields)}

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

    @solver_process('spherical_dens_temp', step=2, args=['gravitational_potential'], kwargs=dict(create_field=True,
                                                                                                 add_gradient_field=True))
    @solver_process('homoeoidal_dens_temp', step=2, args=['gravitational_potential'], kwargs=dict(create_field=True,
                                                                                                  add_gradient_field=True))
    @solver_process('spherical_dens_tden', step=4, args=['pressure'], kwargs=dict(create_field=True,
                                                                                                 add_gradient_field=False))
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

        print(radii)

        other_field_spline = self._get_radial_spline(other_field)
        density_function = self.profiles['gas_density']

        # Integrate HSE field
        hse_field = self._integrate_hse_radial(field_name, other_field_spline, density_function, radii)

        # Add the HSE field to the field container if requested
        hse_field = self._assign_units_and_add_field(field_name, hse_field, create_field, axes=['r'])

        # Compute gradient field if requested
        if add_gradient_field:
            gradient_field = self._compute_gradient_hse(
                field_name, other_field_spline, gradient_field_name, create_field
            )
            return hse_field, gradient_field

        return hse_field

    @solver_process('spherical_dens_temp', step=1, args=['pressure'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_temp', step=1, args=['pressure'], kwargs=dict(create_field=True))
    @solver_process('homoeoidal_dens_tden', step=5, args=['temperature'], kwargs=dict(create_field=True))
    @solver_process('spherical_dens_tden', step=5, args=['temperature'], kwargs=dict(create_field=True))
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
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(pressure._axes, density._axes)]
            pressure, density = make_grid_fields_broadcastable([pressure[...], density[...]],
                                                               [pressure._axes, density._axes],
                                                               self.coordinate_system)
            field = (pressure * scale_factor) / density
        elif field_name == 'pressure':
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(temperature._axes, density._axes)]
            temperature, density = make_grid_fields_broadcastable([temperature[...], density[...]],
                                                                  [temperature._axes, density._axes],
                                                                  self.coordinate_system)
            field = (temperature * density) / scale_factor
        elif field_name == 'gas_density':
            axes = [ax for ax in self.coordinate_system.AXES if ax in set().union(pressure._axes, temperature._axes)]
            pressure, temperature = make_grid_fields_broadcastable([pressure[...], temperature[...]],
                                                                   [pressure._axes, temperature._axes],
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

    @solver_process('homoeoidal_dens_tden', step=3)
    def solve_poisson_problem_homoeoidal(self):
        # Pull out the coordinates and the density profile from the Model instance.
        coordinates = self.grid_manager.get_coordinates()
        density_profile = self.profiles['total_density']
        _dunits, _lunits = unyt.Unit(density_profile.units), unyt.Unit(self.grid_manager.length_unit)

        # Compute the gravitational potential. This passes down to the poisson solver at the
        # lower level of the coordinate system object.
        # TODO: this should be unifiable using symmetry; however, the Handler object cannot tell
        # that it's a radial handler. This needs to be managed...
        gravitational_potential = self.coordinate_system.solve_radial_poisson_problem(
            density_profile,
            coordinates,
        )
        gravitational_potential = G * unyt.unyt_array(gravitational_potential[..., 0],
                                                      _dunits * _lunits ** 2)  # cut of phi dependence.

        # Assign the data to a the gravitational potential field and proceed.
        self._assign_units_and_add_field('gravitational_potential', gravitational_potential, True, ['r', 'theta'])

    @solver_process('spherical_dens_tden', step=3)
    def solve_poisson_problem_spherical(self):
        # Pull out the coordinates and the density profile from the Model instance.
        coordinates = self._get_radial_coordinates()
        density_profile = self.profiles['total_density']
        _dunits, _lunits = unyt.Unit(density_profile.units), unyt.Unit(self.grid_manager.length_unit)

        # Compute the gravitational potential. This passes down to the poisson solver at the
        # lower level of the coordinate system object.
        # TODO: this should be unifiable using symmetry; however, the Handler object cannot tell
        # that it's a radial handler. This needs to be managed...
        # noinspection PyUnresolvedReferences
        gravitational_potential = self.coordinate_system.solve_radial_poisson_problem(
            density_profile,
            coordinates.d,
        )
        gravitational_potential = G * unyt.unyt_array(gravitational_potential,
                                                      _dunits * _lunits ** 2)  # cut of phi dependence.

        # Assign the data to a the gravitational potential field and proceed.
        self._assign_units_and_add_field('gravitational_potential', gravitational_potential, True, ['r'])

    @solver_process('homoeoidal_dens_tden', step=4)
    def solve_hse_asymmetric(self):
        # Pull the coordinates and the units necessary. Because we are integrating radially, only
        # the radii matter, not the angular coordinates.
        radius = self._get_radial_coordinates()
        print(radius)
        gas_density_profile = self.profiles['gas_density']
        gravitational_potential = self.FIELDS['gravitational_potential'][...]

        # Build the splines that are necessary for our integration scheme.
        # - gas_density_spline: rho -- we only need this for the derivative.
        # - middle_integrand_spline: d(rho)/dr * phi
        gas_density_spline = InterpolatedUnivariateSpline(radius.d, gas_density_profile(radius.d))

        # construct the middle_integrand_spline
        deriv_gas_dens = gas_density_spline(radius.d, 1)[..., np.newaxis]  # -> reshaped to ensure broadcastability
        middle_integrand_spline = make_interp_spline(radius.d, deriv_gas_dens * gravitational_potential.d)

        # Compute the integrals for the pressure.
        # 1. -rho(r)*phi(r)
        # 2. - int_r^r_max d(rho)/dr * phi(r) dr
        # 3. - int_r_max^inf d(rho)/dr * phi(r) dr.
        extended_potential = lambda _r: gravitational_potential[-1, :].d * (_r / radius.d[-1]) ** (-1)
        middle_integrand = lambda _r: middle_integrand_spline(_r)
        outer_integrand = lambda _r: gas_density_spline(_r, 1) * extended_potential(_r)

        # compute the relevant integrals
        inner_integral = gas_density_spline(radius.d)[..., np.newaxis] * gravitational_potential.d
        middle_integral = integrate_vectorized(middle_integrand, radius.d, x_0=radius.d[-1])
        outer_integral = quad_vec(outer_integrand, radius.d[-1], np.inf)[0]

        # Set the pressure and coerce the units
        pressure = -inner_integral - middle_integral  # TODO: the boundary needs to be dealt with.
        base_units = self.FIELDS['gravitational_potential'].units * self.FIELDS['gas_density'].units
        pressure = unyt.unyt_array(pressure, base_units)

        # Pass to the unit setter and field adder.
        self._assign_units_and_add_field('pressure', pressure, True, ['r', 'theta'])


if __name__ == '__main__':
    from pisces.profiles import NFWDensityProfile, IsothermalTemperatureProfile
    from pisces.geometry import SphericalCoordinateSystem, OblateHomoeoidalCoordinateSystem

    print(ClusterModel.list_pathways())
    d = NFWDensityProfile(rho_0=1e5, r_s=100)
    td = NFWDensityProfile(rho_0=1e6, r_s=150)
    t = IsothermalTemperatureProfile(T_0=5)

    cs_sphere = SphericalCoordinateSystem()
    cs_ho = OblateHomoeoidalCoordinateSystem(ecc=0.999)

    model_hom = ClusterModel.from_dens_and_tden('test_hom.hdf5', 1e-1, 1e4, d, td, coordinate_system=cs_ho, n_theta=50,
                                                overwrite=True)
    model_s = ClusterModel.from_dens_and_tden('test_s.hdf5', 1e-1, 1e4, d, td, coordinate_system=cs_sphere,
                                                overwrite=True)
    # model = ClusterModel.from_dens_and_temp('test.hdf5', 1e-1, 1e4, d, t, coordinate_system=cs, overwrite=True)
    import matplotlib.pyplot as plt

    r = model_s._get_radial_coordinates().d
    plt.loglog(r, model_hom.FIELDS['total_mass'][:])
    plt.loglog(r, model_s.FIELDS['total_mass'][:])
    plt.show()
