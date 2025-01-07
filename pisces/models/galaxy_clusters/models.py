"""
Pisces model classes for describing galaxy clusters.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import unyt
from numpy.typing import ArrayLike
from scipy.integrate import quad, quad_vec
from scipy.interpolate import InterpolatedUnivariateSpline, make_interp_spline

from pisces.geometry import CoordinateSystem, SphericalCoordinateSystem
from pisces.io import HDF5_File_Handle
from pisces.models.base import _RadialModel
from pisces.models.galaxy_clusters.grids import ClusterGridManager
from pisces.models.solver import (
    serial_solver_checkers,
    serial_solver_processes,
    solver_process,
)
from pisces.models.utilities import ModelConfigurationDescriptor
from pisces.utilities.math_utils.numeric import integrate, integrate_vectorized
from pisces.utilities.physics import G, m_p, mu

if TYPE_CHECKING:
    from pisces.geometry.base import RadialCoordinateSystem
    from pisces.profiles.base import Profile


class ClusterModel(_RadialModel):
    r""" """
    # @@ VALIDATION MARKERS @@ #
    # These validation markers are used by the Model to constrain the valid
    # parameters for the model. Subclasses can modify the validation markers
    # to constrain coordinate system compatibility.
    #
    # : _IS_ABC : marks whether the model should seek out pathways or not.
    # : _INHERITS_PATHWAYS: will allow subclasses to inherit the pathways of their parent class.
    _IS_ABC: bool = False
    _INHERITS_PATHWAYS: bool = False

    # @@ CLASS PARAMETERS @@ #
    # The class parameters define several "standard" behaviors for the class.
    # These can be altered in subclasses to produce specific behaviors.
    GRID_MANAGER_TYPE = ClusterGridManager
    config = ModelConfigurationDescriptor(filename="galaxy_clusters.yaml")

    # @@ CONSTRUCTION METHODS @@ #
    # `build_skeleton` gets overwritten here because we want to implement the gcluster
    # specific norms for the bbox and grid spacing.
    # noinspection PyMethodOverriding
    @classmethod
    def build_skeleton(
        cls,
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
        profiles: Optional[Dict[str, "Profile"]] = None,
        coordinate_system: Optional[SphericalCoordinateSystem] = None,
    ) -> "HDF5_File_Handle":
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
            coordinate_system = cls.DEFAULT_COORDINATE_SYSTEM(
                **cls.DEFAULT_COORDINATE_SYSTEM_PARAMS
            )
        cls._cls_validate_coordinate_system(coordinate_system)

        # Determine requirements based on coordinate system
        requires_theta = coordinate_system.__class__.__name__ not in [
            "SphericalCoordinateSystem"
        ]
        requires_phi = coordinate_system.__class__.__name__ not in [
            "OblateHomoeoidalCoordinateSystem",
            "ProlateHomoeoidalCoordinateSystem",
            "SphericalCoordinateSystem",
        ]

        # Ensure that the user has provided n_phi and n_theta if they are required.
        # Otherwise, we return an error.
        n_phi = n_phi or (1 if not requires_phi else None)
        if n_phi is None:
            raise ValueError(
                f"Parameter `n_phi` is required for coordinate systems "
                f"of type {coordinate_system.__class__.__name__}."
            )
        n_theta = n_theta or (1 if not requires_theta else None)
        if n_theta is None:
            raise ValueError(
                f"Parameter `n_theta` is required for coordinate systems "
                f"of type {coordinate_system.__class__.__name__}."
            )

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
    def from_dens_and_tden(
        cls,
        path: str,
        r_min: float,
        r_max: float,
        gas_density: "Profile",
        total_density: "Profile",
        num_points: Optional[int] = 1000,
        n_phi: Optional[float] = None,
        n_theta: Optional[float] = None,
        chunk_shape: Union[int, ArrayLike] = None,
        extra_profiles: Optional[Dict[str, "Profile"]] = None,
        coordinate_system: Optional[CoordinateSystem] = None,
        **kwargs,
    ) -> "ClusterModel":
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
        profiles = {"gas_density": gas_density, "total_density": total_density}
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
        if coordinate_system_name == "SphericalCoordinateSystem":
            obj(pathway="spherical_dens_tden")
        elif "Homoeoidal" in coordinate_system_name:
            obj(pathway="homoeoidal_dens_tden")
        else:
            raise NotImplementedError(
                f"The coordinate system {coordinate_system_name} is an accepted coordinate"
                f" system for {cls.__name__}, but there is no density / total density pipeline implemented."
            )

        return obj

    @classmethod
    def from_dens_and_temp(
        cls,
        path: str,
        r_min: float,
        r_max: float,
        /,
        gas_density: "Profile",
        temperature: "Profile",
        *,
        num_points: Optional[int] = 1000,
        n_phi: Optional[float] = None,
        n_theta: Optional[float] = None,
        chunk_shape: Union[int, ArrayLike] = None,
        extra_profiles: Optional[Dict[str, "Profile"]] = None,
        coordinate_system: Optional[CoordinateSystem] = None,
        **kwargs,
    ) -> "ClusterModel":
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
        profiles = {"gas_density": gas_density, "temperature": temperature}
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
        if obj.coordinate_system.__class__.__name__ == "SphericalCoordinateSystem":
            obj(pathway="spherical_dens_temp")
        elif "Homoeoidal" in obj.coordinate_system.__class__.__name__:
            obj(pathway="homoeoidal_dens_temp")
        else:
            raise NotImplementedError(
                f"The coordinate system {obj.coordinate_system.__class__.__name__} is an accepted coordinate"
                f" system for {cls.__name__}, but there is no density / temperature pipeline implemented."
            )

        return obj

    @property
    def coordinate_system(self) -> "RadialCoordinateSystem":
        # noinspection PyTypeChecker
        # skip the type checking because we have enforcements on the
        # coordinate system which are dynamic.
        return super().coordinate_system

    # @@ UTILITY FUNCTIONS @@ #
    # These utility functions are used throughout the model generation process for various things
    # and are not sufficiently general to be worth implementing elsewhere.
    def _integrate_hse_radial(
        self,
        target_field: str,
        other_field_spline: InterpolatedUnivariateSpline,
        density_function: callable,
        radii: unyt.unyt_array,
    ) -> unyt.unyt_array:
        r"""
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
        if target_field == "gravitational_potential":
            # The other field is the pressure, dphi = dP/rho
            integrand = lambda _r: other_field_spline(_r, 1) / density_function(_r)
            result_units = (
                self.FIELDS["pressure"].units / self.FIELDS["gas_density"].units
            )
        elif target_field == "pressure":
            # The other field is potential, dP = dphi*rho
            integrand = lambda _r: other_field_spline(_r, 1) * density_function(_r)
            result_units = (
                self.FIELDS["gravitational_potential"].units
                * self.FIELDS["gas_density"].units
            )
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
        if target_field == "gravitational_potential":
            integrated_field -= integrand(radii.d[-1]) * radii.d[-1]
        elif target_field == "pressure":
            boundary_integral = (
                other_field_spline(radii.d[-1], 1) * radii.d[-1] ** 2
            ) * quad(lambda _r: density_function(_r) / _r**2, radii.d[-1], np.inf)[0]
            integrated_field += boundary_integral
        else:
            raise ValueError(f"Target field '{target_field}' not recognized.")

        # Return the output as an array with the result units.
        return unyt.unyt_array(integrated_field, result_units)

    def _compute_gradient_hse(
        self,
        target_field: str,
        other_field_spline: InterpolatedUnivariateSpline,
        gradient_field_name: str,
        create_field: bool,
    ) -> unyt.unyt_array:
        r"""
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
        gradient_axes = self.geometry_handler.get_gradient_dependence(axes=["r"])
        coordinates = self.grid_manager.get_coordinates(axes=gradient_axes)

        # Compute the gradient of the spline that was provided as the other field spline. This
        # will be either the gravitational potential spline or the pressure spline.
        # NOTE that we take the zero component because the returned grid is (..., 1) in shape -> (...,) in final
        # shape.
        gradient = self.geometry_handler.compute_gradient(
            other_field_spline,
            coordinates,
            axes=["r"],
            derivatives=[lambda _r: other_field_spline(_r, 1)],
            basis="unit",
        )[..., 0]

        # Ensure that the fields are broadcastable to one another. This is
        # required if the the gradient breaks symmetry and has a larger dimension that
        # the density field does.
        gradient, density = self.grid_manager.make_fields_consistent(
            [gradient, self.FIELDS["gas_density"][...]], [gradient_axes, ["r"]]
        )

        # Determine units -> This is a simple application of the
        # HSE equation to determine dimensionality. Use the length unit to manage
        # the unit manipulations from differential operations.
        if target_field == "pressure":
            gradient_field = gradient * density
            gradient_units = (
                self.FIELDS["gravitational_potential"].units
                * self.FIELDS["gas_density"].units
            ) / unyt.Unit(self.grid_manager.length_unit)
        elif target_field == "gravitational_potential":
            gradient_field = gradient / density
            gradient_units = self.FIELDS["pressure"].units / (
                self.FIELDS["gas_density"].units
                * unyt.Unit(self.grid_manager.length_unit)
            )
        else:
            raise ValueError("Unknown target field.")

        # Enforce units and register the new field.
        gradient_field = unyt.unyt_array(gradient_field, gradient_units)
        return self._assign_default_units_and_add_field(
            gradient_field_name, gradient_field, create_field, axes=gradient_axes
        )

    # @@ CHECKERS @@ #
    # There is only one checker for the galaxy cluster pathways.
    @serial_solver_checkers(
        [
            "spherical_dens_temp",
            "spherical_dens_tden",
            "homoeoidal_dens_tden",
            "homoeoidal_dens_temp",
        ]
    )
    def check_pathways(self, pathway: str) -> bool:
        r"""
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
        if pathway.startswith("spherical"):
            state = state and (cs_name == "SphericalCoordinateSystem")
        if pathway.startswith("homoeidal"):
            state = state and ("Homoeoidal" in cs_name)

        # CHECKING profiles
        # We must have the correct input fields to proceed.
        if pathway.endswith("dens_temp"):
            state = state and ("temperature" in self.profiles)
            state = state and ("gas_density" in self.profiles)
        elif pathway.endswith("dens_tden"):
            state = state and ("total_density" in self.profiles)
            state = state and ("gas_density" in self.profiles)

        return state

    # @@ SOLVER PROCESSES @@ #
    # All of the pipelines are implemented below.
    @serial_solver_processes(
        [
            ("spherical_dens_temp", 0, [], {}),
            ("spherical_dens_tden", 0, [], {}),
            ("homoeoidal_dens_temp", 0, [], {}),
            ("homoeoidal_dens_tden", 0, [], {}),
        ]
    )
    def convert_profiles_to_fields(self):
        r"""
        Convert the registered model profiles into fields with the relevant axes.

        This solver process ensures that every registered profile in :py:attr:`profiles` gets converted
        to a matching field in :py:attr:`FIELDS`. Additionally, this method will ensure that the
        ``stellar_density`` field is initialized as an empty field if it is not provided in the profiles.

        .. hint::

            The reason for this is to ensure that we don't need special logic to manage the fact that
            it's missing down the line.

        .. rubric:: Pathways Summary

        +---------------------------+-------------------+
        | Pathway                   |  Step #           |
        +===========================+===================+
        | ``'spherical_dens_temp'`` | ``0``             |
        +---------------------------+-------------------+
        | ``'spherical_dens_tden'`` | ``0``             |
        +---------------------------+-------------------+
        |``'homoeoidal_dens_temp'`` | ``0``             |
        +---------------------------+-------------------+
        |``'homoeoidal_dens_temp'`` | ``0``             |
        +---------------------------+-------------------+

        """
        # Cycle through all of the profiles and add them to the
        # field set for the model.
        for _profile in self.profiles.keys():
            self.convert_profile_to_field(_profile)

        # Continue by adding the stellar density if it's not already present.
        # This ensures that it can be used in computations elsewhere.
        if "stellar_density" not in self.FIELDS:
            _units = self.get_default_units("stellar_density")
            self.logger.debug(
                "[EXEC] \t\tAdded field `stellar_density` (units=%s) as null.",
                str(_units),
            )
            self.FIELDS.add_field(
                "stellar_density",
                axes=["r"],
                units=str(_units),
            )

    @serial_solver_processes(
        [
            ("homoeoidal_dens_temp", 6, [["gas_density", "stellar_density"]], {}),
            (
                "spherical_dens_temp",
                6,
                [["gas_density", "stellar_density", "dark_matter_density"]],
                {},
            ),
            (
                "homoeoidal_dens_tden",
                2,
                [
                    [
                        "gas_density",
                        "stellar_density",
                        "dark_matter_density",
                        "total_density",
                    ]
                ],
                {},
            ),
            (
                "spherical_dens_tden",
                2,
                [
                    [
                        "gas_density",
                        "stellar_density",
                        "dark_matter_density",
                        "total_density",
                    ]
                ],
                {},
            ),
        ]
    )
    def integrate_density_fields(
        self, density_field_names: List[str], create_fields: bool = True, overwrite=True
    ):
        r"""
        Integrate radial density profiles to compute corresponding mass profiles.

        This solver process takes specified radial density fields (e.g., gas, stellar, dark matter density),
        integrates them over the radial grid, and computes the enclosed mass profiles. The resulting mass
        profiles are returned as ``unyt.unyt_array`` objects. Optionally, the integrated mass fields can
        be added to the model's field container.

        .. rubric:: Pathways Summary

        +---------------------------+--------+-------------------------------------------------------------------+
        | Pathway                   | Step # | Profiles Integrated                                               |
        +===========================+========+===================================================================+
        | ``'homoeoidal_dens_temp'``| ``6``  | :math:`\rho_g, \; \rho_\star`                                     |
        +---------------------------+--------+-------------------------------------------------------------------+
        | ``'spherical_dens_temp'`` | ``6``  | :math:`\rho_g, \; \rho_\star,\; \rho_{dm}`                        |
        +---------------------------+--------+-------------------------------------------------------------------+
        | ``'homoeoidal_dens_tden'``| ``2``  | :math:`\rho_g, \; \rho_\star, \; \rho_{\rm dm}, \; \rho{\rm dyn}` |
        +---------------------------+--------+-------------------------------------------------------------------+
        | ``'spherical_dens_tden'`` | ``2``  | :math:`\rho_g, \; \rho_\star, \; \rho_{\rm dm}, \; \rho{\rm dyn}` |
        +---------------------------+--------+-------------------------------------------------------------------+

        Parameters
        ----------
        density_field_names : List[str]
            List of names of the density fields to integrate. Permitted fields include:
            ``['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']``.

        create_fields : bool, optional
            If True, the integrated mass fields will be added to the grid manager's field container.
            Defaults to True.

        overwrite : bool, optional
            If True, existing mass fields with the same name will be overwritten. If False and the field
            already exists, a ``ValueError`` is raised. Defaults to False.

        Returns
        -------
        Tuple[unyt.unyt_array, ...]
            A tuple of integrated mass profiles as ``unyt.unyt_array`` objects.

        Raises
        ------
        ValueError
            If any density field name is invalid or not defined over the radial ('`r'`) axis.

        KeyError
            If a corresponding mass field name cannot be determined from the configuration.

        Notes
        -----
        This method assumes that all density fields provided are defined over radial coordinates ('`r'` axis).
        It utilizes the coordinate system's :py:meth:`integrate_radial_density_field` method to perform the integration,
        ensuring compatibility with the cluster's geometry and physics.

        """
        # Define valid density fields for validation. Fields not in the valid set are checked for errors.
        valid_density_fields = [
            "total_density",
            "gas_density",
            "stellar_density",
            "dark_matter_density",
        ]

        returned_fields = []
        for density_field in density_field_names:
            if density_field not in valid_density_fields:
                raise ValueError(
                    f"The density field `{density_field}` is not among the valid density fields"
                    " for this model."
                )

            returned_fields.append(
                self.integrate_radial_density_field(
                    density_field,
                    create_field=create_fields,
                    overwrite=overwrite,
                )
            )

    @serial_solver_processes(
        [
            (
                "homoeoidal_dens_temp",
                5,
                ["dark_matter_density"],
                dict(create_field=True),
            ),
            (
                "homoeoidal_dens_temp",
                7,
                ["dark_matter_mass"],
                dict(create_field=True, field_type="mass"),
            ),
            (
                "spherical_dens_temp",
                5,
                ["dark_matter_density"],
                dict(create_field=True),
            ),
            (
                "homoeoidal_dens_tden",
                1,
                ["dark_matter_density"],
                dict(create_field=True),
            ),
            (
                "spherical_dens_tden",
                1,
                ["dark_matter_density"],
                dict(create_field=True),
            ),
        ]
    )
    def perform_mass_accounting(
        self,
        target_field: str,
        create_field: bool = False,
        units: Union[str, unyt.Unit] = None,
        mode: str = "density",
    ):
        r"""
        Perform mass accounting by computing a target mass or density field.

        This solver process calculates a specified mass or density field by summing or subtracting
        existing component fields. It ensures that the target field accurately reflects the total or
        remaining mass/density based on the provided mode.

        .. rubric:: Pathways Summary

        +----------------------------+--------+--------------------------------------------+
        | Pathway                    | Step # | Fields Involved                            |
        +============================+========+============================================+
        | ``'homoeoidal_dens_temp'`` |``5,7`` | :math:`\rho_{\rm dm}, M_{\rm dm}`          |
        +----------------------------+--------+--------------------------------------------+
        | ``'spherical_dens_temp'``  | ``5``  | :math:`\rho_{\rm dm}`                      |
        +----------------------------+--------+--------------------------------------------+
        | ``'homoeoidal_dens_tden'`` | ``1``  | :math:`\rho_{\rm dm}`                      |
        +----------------------------+--------+--------------------------------------------+
        | ``'spherical_dens_tden'``  | ``1``  | :math:`\rho_{\rm dm}`                      |
        +----------------------------+--------+--------------------------------------------+

        Parameters
        ----------
        target_field : str
            The name of the field to compute. Must be one of the valid density or mass fields:
            - For density: ``['total_density', 'gas_density', 'stellar_density', 'dark_matter_density']``
            - For mass: ``['total_mass', 'gas_mass', 'stellar_mass', 'dark_matter_mass']``

        create_field : bool, default=False
            If True, the computed field will be added to the model's field container (`self.FIELDS`).
            Defaults to False.

        units : str or unyt.Unit, optional
            The desired units for the target field. If not provided, the method retrieves the default
            units for `target_field` from the model's configuration.

        mode : str, default='density'
            The type of accounting to perform. Must be either:
            - `'density'`: Compute a density field by summing or subtracting density components.
            - `'mass'`: Compute a mass field by summing or subtracting mass components.

        Returns
        -------
        unyt.unyt_array
            The computed mass or density field with appropriate units.

        Raises
        ------
        ValueError
            - If `mode` is not `'density'` or `'mass'`.
            - If `target_field` is not among the valid fields for the specified `mode`.
            - If required component fields are missing from `self.FIELDS`.

        KeyError
            - If the default units for `target_field` cannot be determined from the configuration.

        See Also
        --------
        integrate_radial_density_field : Integrates a single radial density field to compute the mass profile.
        convert_profile_to_field : Converts a single profile to a field.
        get_default_units : Retrieves the default units for a given field.
        FIELDS.add_field : Adds a new field to the model's field container.
        """
        # Validate the target field and field_type. The desired field must be a specified valid field
        # and we need to have access to all the additional components that are necessary to perform
        # the computation further along in the method.
        if mode == "density":
            valid_fields = [
                "total_density",
                "gas_density",
                "stellar_density",
                "dark_matter_density",
            ]
        elif mode == "mass":
            valid_fields = [
                "total_mass",
                "gas_mass",
                "stellar_mass",
                "dark_matter_mass",
            ]
        else:
            raise ValueError("Invalid `field_type`. Must be 'density' or 'mass'.")

        # Check that the target field is among the valid fields
        if target_field not in valid_fields:
            raise ValueError(
                f"'{target_field}' is not a valid {mode} field. "
                f"Valid fields are: {valid_fields}."
            )

        # Check that we have all of the necessary alternative fields
        _required_fields = [_fld for _fld in valid_fields if (_fld != target_field)]
        _missing_required_fields = [
            _fld for _fld in _required_fields if _fld not in self.FIELDS
        ]
        if len(_missing_required_fields) > 0:
            raise ValueError(
                f"Failed to perform mass accounting (mode={mode}) to compute {target_field}:\n"
                f"Some required fields are not available: {_missing_required_fields}."
            )

        # Coerce units and ensure that we can retrieve units for the fields.
        # If we didn't get given units for the field, we'll look up the target field to get the units.
        if units is None:
            units = self.get_default_units(target_field)

        # Retrieve the fields necessary for the computation. This requires grabbing all of the required fields
        # that we have already generated and getting access to the underlying array. Once the fields are
        # retrieved, they need to be coerced to be self-consistent dimensionally.
        # !NOTE the computation_fields are just references, not arrays.
        computation_fields = {
            field_name: self.FIELDS[field_name] for field_name in _required_fields
        }
        _comp_field_arrays, _comp_field_axes = zip(
            *((_f[...].to_value(units), _f.AXES) for _f in computation_fields.values())
        )
        axes_set = self.coordinate_system.ensure_axis_order(
            set().union(*list(_comp_field_axes))
        )
        computation_fields = self.grid_manager.make_fields_consistent(
            list(_comp_field_arrays),
            list(_comp_field_axes),
        )

        # !NOTE the computational fields are consistent but not stackable yet. We need to now
        # broadcast them to stackable shapes and figure out what the output shape / axes will be.
        axes_mask = self.coordinate_system.build_axes_mask(axes_set)
        target_grid_shape = self.grid_manager.GRID_SHAPE[axes_mask]
        computation_fields = [
            np.broadcast_to(_fld, target_grid_shape) for _fld in computation_fields
        ]

        # Perform the accounting operation. If we are computing a total field, we simply need
        # to add. Otherwise we need to pull out the total field and subtract the get the final
        # value.
        if target_field == valid_fields[0]:
            # We are computing a total field, not a component field.
            output_field = unyt.unyt_array(
                np.sum(np.stack(computation_fields, axis=0), axis=0), units
            )
        else:
            output_field = computation_fields[0] - np.sum(
                np.stack(computation_fields[1:], axis=0), axis=0
            )
            output_field = unyt.unyt_array(output_field, units)

        # Add the resulting field as a new field in the model if we are told to do so.
        output_field = self._assign_default_units_and_add_field(
            target_field, output_field, create_field, axes=axes_set
        )
        return output_field

    @serial_solver_processes(
        [
            ("spherical_dens_temp", 1, ["pressure"], dict(create_field=True)),
            ("homoeoidal_dens_temp", 1, ["pressure"], dict(create_field=True)),
            ("homoeoidal_dens_tden", 5, ["temperature"], dict(create_field=True)),
            ("spherical_dens_tden", 5, ["temperature"], dict(create_field=True)),
        ]
    )
    def solve_eos(self, target_field: str, create_field: bool = False):
        r"""
        Solve the Equation of State (EoS) for thermodynamic fields.

        This solver process computes the specified ``target_field`` (``'pressure'``, ``'temperature'``, or ``'gas_density'``)
        based on its dependencies. Optionally, the computed field can be added to the model's field container.

        .. note::

            This method assumes an **ideal gas** equation of state.


        Parameters
        ----------
        target_field : str
            The thermodynamic field to compute. Must be one of:

            - ``'pressure'``
            - ``'temperature'``
            - ``'gas_density'``

        create_field : bool, default=False
            If ``True``, add the computed field to the model's field container (:py:attr:`FIELDS`).

        Returns
        -------
        unyt.unyt_array
            The computed thermodynamic field with appropriate units.

        Raises
        ------
        ValueError
            - If ``target_field`` is not one of ``'pressure'``, ``'temperature'``, or ``'gas_density'``.
            - If required dependent fields are missing.

        KeyError
            - If default units for ``target_field`` cannot be determined from the configuration.


        Notes
        -----

        - **Field Dependencies**:

          - ``'temperature'``: Requires ``'pressure'`` and ``'gas_density'``.
          - ``'pressure'``: Requires ``'temperature'`` and ``'gas_density'``.
          - ``'gas_density'``: Requires ``'pressure'`` and ``'temperature'``.

        - **Equation of State**:

          - For ``'temperature'``:

            .. math::
                T = \frac{m_p \mu \cdot P}{\rho}

          - For ``'pressure'``:

            .. math::
                P = \frac{T \cdot \rho}{m_p \mu}

          - For ``'gas_density'``:

            .. math::
                \rho = \frac{m_p \mu \cdot P}{T}

          Where:

          - :math:`P` is pressure,
          - :math:`T` is temperature,
          - :math:`\rho` is gas density,
          - :math:`m_p` is the proton mass,
          - :math:`\mu` is the mean molecular weight.

        """
        scale_factor = m_p * mu  # Universal scale factor for the equation of state.
        # Validate the target field and obtain the necessary data. Each field has two dependencies which need to be
        # checked. All fields must be present to proceed. Additionally obtain the base units and target units etc.
        required_fields = {
            "temperature": ["pressure", "gas_density"],
            "pressure": ["temperature", "gas_density"],
            "gas_density": ["pressure", "temperature"],
        }
        self._validate_field_dependencies(target_field, required_fields)

        # Pull the fields we want out of the FIELDS attribute.
        # we use the required fields dictionary to figure out which ones to pull.
        # !NOTE these are just field references.
        input_field_A, input_field_B = tuple(
            [self.FIELDS[_in_field] for _in_field in required_fields[target_field]]
        )
        units_A, units_B = input_field_A.units, input_field_B.units
        output_field_axes = self.coordinate_system.ensure_axis_order(
            set().union(input_field_A.AXES, input_field_B.AXES)
        )

        # Ensure that the input fields are broadcastable to one another.
        input_field_A, input_field_B = tuple(
            self.grid_manager.make_fields_consistent(
                [input_field_A[...].d, input_field_B[...].d],
                [input_field_A.AXES, input_field_B.AXES],
            )
        )

        # Perform the base-level computation and enforce the base units.
        # The units are later coerced based on our knowledge of the base units.
        if target_field == "temperature":
            # A: pressure, B: gas_density, operation: (m_p mu)*(P/rho)
            output_field = scale_factor.d * (input_field_A / input_field_B)
            output_units = scale_factor.units * (units_A / units_B)
        elif target_field == "pressure":
            # A: T, B: rho, operation: (T*rho)/(m_p mu)
            output_field = (input_field_A * input_field_B) / scale_factor.d
            output_units = (units_A * units_B) / scale_factor
        elif target_field == "gas_density":
            # A: pressure, B: temperature, operation: (m_p mu)*(P/T)
            output_field = scale_factor.d * (input_field_A / input_field_B)
            output_units = scale_factor.units * (units_A / units_B)
        else:
            raise ValueError(
                "Invalid `target_field`. This should never happen due to validation earlier in the method!"
            )

        output_field = unyt.unyt_array(output_field, output_units)
        # Set units and create the field if required
        field = self._assign_default_units_and_add_field(
            target_field, output_field, create_field, output_field_axes
        )
        return field

    @serial_solver_processes(
        [
            (
                "spherical_dens_temp",
                2,
                ["gravitational_potential"],
                dict(create_field=True, add_gradient_field=True),
            ),
            (
                "homoeoidal_dens_temp",
                2,
                ["gravitational_potential"],
                dict(create_field=True),
            ),
            (
                "spherical_dens_tden",
                4,
                ["pressure"],
                dict(create_field=True, add_gradient_field=False),
            ),
        ]
    )
    def solve_hse(
        self,
        target_field: str,
        create_field: bool = False,
        add_gradient_field: bool = False,
    ):
        r"""
        Solve the Hydrostatic Equilibrium (HSE) equation to compute a thermodynamic or potential field.

        This solver process computes the specified ``target_field`` (either ``'gravitational_potential'`` or ``'pressure'``)
        using the HSE equation, which balances gravitational forces and pressure gradients. Optionally, the computed field's
        gradient can also be added to the model's field container.

        .. math::

            \frac{dP}{dr} = -\rho \frac{d\Phi}{dr}

        Parameters
        ----------
        target_field : str
            The field to compute. Must be one of:

            - ``'gravitational_potential'``: Computes the gravitational potential field.
            - ``'pressure'``: Computes the pressure field.

        create_field : bool, default=False
            If ``True``, the computed field is added to the model's field container (:py:attr:`FIELDS`).

        add_gradient_field : bool, default=False
            If ``True``, the gradient of the computed field is also added to the model's field container.

        Returns
        -------
        unyt.unyt_array or tuple of unyt.unyt_array
            The computed field (and its gradient, if ``add_gradient_field=True``) with appropriate units.

        Raises
        ------
        ValueError
            If ``target_field`` is not one of ``'gravitational_potential'`` or ``'pressure'``.
        KeyError
            If the required dependent fields are not present in the model.

        Notes
        -----
        .. rubric:: Pathways Summary

        +----------------------------+--------+-----------------------------------------------+
        | Pathway                    | Step # | Fields Involved                               |
        +============================+========+===============================================+
        | ``'spherical_dens_temp'``  |   2    | :math:`\nabla \Phi,\;\Phi`                    |
        +----------------------------+--------+-----------------------------------------------+
        | ``'homoeoidal_dens_temp'`` |   2    | :math:`\nabla \Phi,\;\Phi`                    |
        +----------------------------+--------+-----------------------------------------------+
        | ``'spherical_dens_tden'``  |   4    | :math:`P`                                     |
        +----------------------------+--------+-----------------------------------------------+

        """
        valid_fields = {
            "gravitational_potential": ("pressure", "pressure_gradient"),
            "pressure": ("gravitational_potential", "gravitational_field"),
        }

        # Validate target fields and determine the output field name, gradient field name, and other core
        # parameters of the method.
        if target_field not in valid_fields:
            raise ValueError(
                f"Invalid target_field '{target_field}'. Must be one of {valid_fields}."
            )
        input_field_name, input_grad_field_name = valid_fields[target_field]

        # Setup the necessary structure for the computation. Obtain the splines and radii along with
        # the density profile function.
        radii = self.get_radii()
        input_field_spline = self.construct_radial_spline(input_field_name)
        gas_density_function = self.profiles["gas_density"]

        # Perform the HSE integral radially. This will automatically determine which way the equation
        # manifests and then perform the integration in correct units.
        hse_field = self._integrate_hse_radial(
            target_field, input_field_spline, gas_density_function, radii
        )
        hse_field = self._assign_default_units_and_add_field(
            target_field, hse_field, create_field, axes=["r"]
        )

        # If we've been asked to include the gradient (as its own field), we need to
        # do that secondary computation. Passes off to self._compute_gradient_hse to perform the
        # operation.
        if add_gradient_field:
            gradient_field = self._compute_gradient_hse(
                target_field, input_field_spline, input_grad_field_name, create_field
            )
            return hse_field, gradient_field

        return hse_field

    @serial_solver_processes(
        [("homoeoidal_dens_tden", 3, [], {}), ("spherical_dens_tden", 3, [], {})]
    )
    def solve_poisson_problem(self):
        # Validation step. Ensure that we have a valid coordinate and manually determine the
        # necessary axes for the computation (both as input and output fields).
        _coordinate_system_name = self.coordinate_system.__class__.__name__
        if _coordinate_system_name == "SphericalCoordinateSystem":
            # We're going to solve the poisson problem in 1D spherical coordinates. We only need
            # the radii and the density profile.
            coordinates = self.get_radii().d  # -> ensure we convert to a plain array.
            output_axes = ["r"]
        elif _coordinate_system_name in [
            "OblateHomoeoidalCoordinateSystem",
            "ProlateHomoeoidalCoordinateSystem",
            "PseudoSphericalCoordinateSystem",
        ]:
            # We're going to solve in ellipsoidal coordinates. We need all coordinates (even if redundant) because
            # the solver comes from the base class PseudoSphericalCoordinateSystem.
            # TODO: maybe this behavior can be refined later?
            coordinates = self.grid_manager.get_coordinates(
                axes=["r", "theta", "phi"]
            )  # See above comment.
            output_axes = ["r", "theta"]

            if _coordinate_system_name == "PseudoSphericalCoordinateSystem":
                output_axes.append("phi")

        else:
            raise ValueError(
                f"The coordinate system `{_coordinate_system_name}` is not supported for Poisson solving."
            )

        # Pull out the density profile and determine the units that are managed in the
        # computation procedure. Because the solve_poisson_problem methods return in Plank units,
        # we need to find the _plank_units_output first and then multiply by G before coercing to
        # final units.
        dynamic_density_profile = self.profiles["total_density"]
        density_base_units, length_base_units = unyt.Unit(
            dynamic_density_profile.units
        ), unyt.Unit(self.grid_manager.length_unit)
        _plank_output_units = density_base_units * (length_base_units**2)

        # Compute the gravitational potential. This passes down to the poisson solver at the
        # lower level of the coordinate system object.
        # noinspection PyUnresolvedReferences
        gravitational_potential = self.coordinate_system.solve_radial_poisson_problem(
            dynamic_density_profile,
            coordinates,
        )

        # Post-processing the gravitational potential: If we have axes to cut down, we need to do so because
        # of the axes constraints. Additionally, units need to be managed correctly.
        if _coordinate_system_name == "SphericalCoordinateSystem":
            gravitational_potential = G * unyt.unyt_array(
                gravitational_potential, _plank_output_units
            )
        elif _coordinate_system_name in [
            "OblateHomoeoidalCoordinateSystem",
            "ProlateHomoeoidalCoordinateSystem",
        ]:
            # We need to cut out the final coordinate dimension of the field.
            gravitational_potential = G * unyt.unyt_array(
                gravitational_potential[..., 0], _plank_output_units
            )
        else:
            # all axes are relevant
            gravitational_potential = G * unyt.unyt_array(
                gravitational_potential, _plank_output_units
            )

        # Assign the data to a the gravitational potential field and proceed.
        self._assign_default_units_and_add_field(
            "gravitational_potential", gravitational_potential, True, output_axes
        )

    # @@ SPECIALITY SOLVERS @@ #
    # These solvers are one-off methods which do very specialized
    # processes that occur in only 1-2 cases across all pathways.
    @solver_process("homoeoidal_dens_temp", step=4)
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
        # Setup the process by fetching the system radii and creating a spline of the
        # gravitational potential.
        # !NOTE: We use the gravitational potential because the field will be 2 or 3D for these
        #   systems since they use non-spherical coordinate systems.
        radius = self.get_radii()
        potential_spline = self.construct_radial_spline("gravitational_potential")

        # Construct the potential derivative and pull the flux factor for the
        # computation.
        # noinspection PyUnresolvedReferences
        # We skip the inspection here because the constrains on the coordinate system are
        # dynamically evaluated.
        flux_factor = self.coordinate_system.flux_parameter
        d_potential_dr = potential_spline(radius.d, 1)  # First derivative wrt radius

        # Compute the numerator for the total mass equation
        numerator = flux_factor * d_potential_dr * radius.d**2
        numerator_units = radius.units * self.FIELDS["gravitational_potential"].units
        numerator = unyt.unyt_array(numerator, numerator_units)

        # Compute total mass using Gauss's theorem
        total_mass = numerator / (4 * np.pi * G)
        self._assign_default_units_and_add_field(
            "total_mass", total_mass, create_field=True, axes=["r"]
        )

    @solver_process("spherical_dens_temp", step=3)
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
        radius = self.get_radii()
        gravitational_field = self.FIELDS["gravitational_field"][...]
        total_mass = (-(radius**2) * gravitational_field) / G
        self._assign_default_units_and_add_field(
            "total_mass", total_mass, create_field=True, axes=["r"]
        )

    @solver_process(
        "spherical_dens_temp",
        step=4,
        args=[["total_density"]],
        kwargs=dict(create_fields=True),
    )
    def compute_densities_from_mass_spherical(
        self, target_fields: List[str], create_fields: bool = False
    ):
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
        output_fields = []
        valid_fields = {
            "total_density": "total_mass",
            "gas_density": "gas_mass",
            "stellar_density": "stellar_mass",
            "dark_matter_density": "dark_matter_mass",
        }

        for field in target_fields:
            if field not in valid_fields:
                raise ValueError(f"Field `{field}` is not valid.")

            output_fields.append(
                self.compute_spherical_density_from_mass(
                    valid_fields[field], field, create_field=create_fields
                )
            )

        return output_fields

    @solver_process("homoeoidal_dens_temp", step=3)
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
        radius = self.get_radii()
        potential_spline = self.construct_radial_spline("gravitational_potential")

        first_derivative_spline = InterpolatedUnivariateSpline(
            radius.d, potential_spline(radius.d, 1)
        )
        second_derivative = lambda r: first_derivative_spline(r, 1)

        # Compute the laplacian
        laplacian_axes = self.geometry_handler.get_laplacian_dependence(axes=["r"])
        coordinates = self.grid_manager.get_coordinates(axes=laplacian_axes)
        laplacian = self.geometry_handler.compute_laplacian(
            potential_spline,
            coordinates,
            axes=["r"],
            first_derivatives=[first_derivative_spline],
            second_derivatives=[second_derivative],
            edge_order=2,
        )

        # Convert Laplacian to unyt array with proper units
        laplacian_units = (
            self.FIELDS["gravitational_potential"].units / radius.units**2
        )
        laplacian = unyt.unyt_array(laplacian, laplacian_units)

        # SCompute total density using the Poisson equation: rho = laplacian / (4 * pi * G)
        total_density = laplacian / (4 * np.pi * G)
        self._assign_default_units_and_add_field(
            "total_density", total_density, create_field=True, axes=laplacian_axes
        )

    @solver_process("homoeoidal_dens_tden", step=4)
    def solve_hse_asymmetric(self):
        # Obtain the necessary coordinates, fields, and profiles. We are integrating radially
        # (but vectorially), so we only need the radii, the gas, and the potential.
        radius = self.get_radii()
        gas_density_profile = self.profiles["gas_density"]
        gravitational_potential = self.FIELDS["gravitational_potential"][...]

        # Build the necessary splines. We only need to create splines of the
        # potential on the grid for each value of theta.
        potential_spline = make_interp_spline(radius.d, gravitational_potential)

        # Construct the integrands. There are only 2 in this scheme:
        # 1. int_r^r_0 rho*dphi/dr dr
        # 2. int_r_0^infty rho * r^-2 dr.
        inner_integrand = lambda _r: potential_spline(_r, 1) * gas_density_profile(_r)
        outer_integrand = lambda _r: gas_density_profile(_r) / _r**2

        # compute the relevant integrals
        inner_integral = integrate_vectorized(
            inner_integrand, radius.d, x_0=radius.d[-1]
        )
        outer_integral = (
            potential_spline(radius.d[-1], 1) * radius.d[-1] ** 2
        ) * quad_vec(outer_integrand, radius.d[-1], np.inf)[0]

        # Set the pressure and coerce the units
        pressure = inner_integral + outer_integral
        base_units = (
            self.FIELDS["gravitational_potential"].units
            * self.FIELDS["gas_density"].units
        )
        pressure = unyt.unyt_array(pressure, base_units)

        # Pass to the unit setter and field adder.
        self._assign_default_units_and_add_field(
            "pressure", pressure, True, ["r", "theta"]
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from pisces.geometry import (
        OblateHomoeoidalCoordinateSystem,
        SphericalCoordinateSystem,
    )
    from pisces.profiles import IsothermalTemperatureProfile, NFWDensityProfile

    print(ClusterModel.list_pathways())
    d = NFWDensityProfile(rho_0=1e5, r_s=10)
    td = NFWDensityProfile(rho_0=5e6, r_s=200)
    t = IsothermalTemperatureProfile(T_0=5)

    cs_sphere = SphericalCoordinateSystem()
    cs_ho = OblateHomoeoidalCoordinateSystem(ecc=0.7)

    # model_hom = ClusterModel.from_dens_and_tden('test_hom.hdf5', 1e-1, 1e4, d, td, coordinate_system=cs_ho, n_theta=50,
    #                                            overwrite=True)

    model_hom = ClusterModel("test_hom.hdf5")
    model_s = ClusterModel("test_s.hdf5")
    plt.semilogx(model_hom.get_radii(), model_hom.FIELDS["temperature"][...])
    plt.semilogx(model_hom.get_radii(), model_s.FIELDS["temperature"][...])
    plt.show()
    # model_hom.plot_slice('pressure',view_axis='x',norm=LogNorm(vmin=1e-22,vmax=1e-17),
    #                   extent=[-1000,1000,-1000,1000],cmap='inferno')
    model_hom.plot_slice(
        "temperature", [-10000, 10000, -10000, 10000], "z", cmap="inferno"
    )

    plt.show()
