"""
Generic skeleton classes for Pisces models.

This module contains the :py:class:`Model` base class.
"""
from pathlib import Path
from typing import Union, List, Dict, Optional, TYPE_CHECKING, Type, Any

from numpy.typing import ArrayLike

from pisces.geometry import GeometryHandler
from pisces.geometry.base import CoordinateSystem
from pisces.io.hdf5 import HDF5_File_Handle
from pisces.models.grids.base import ModelGridManager
from pisces.models.solver import ModelSolver
from pisces.profiles.collections import HDF5ProfileRegistry
from pisces.utilities.logging import LogDescriptor

if TYPE_CHECKING:
    from pisces.profiles.base import Profile
    from logging import Logger
    from pisces.models.grids.base import ModelFieldContainer

# noinspection PyProtectedMember
class _ModelMeta(type):
    """
    Metaclass for dynamically constructing and validating pathways
    for the `Model` class.
    """
    def __new__(cls, name, bases, dct):
        # Initialize the pathways dictionary for the class
        pathways = {}

        # Construct the pathways by examining class attributes
        pathways = cls._construct_pathways(pathways, dct)

        # Validate the constructed pathways
        cls._validate_pathways(pathways)

        # Attach the pathways to the class
        dct["_pathways"] = dict(pathways)

        # Create the class object
        cls_obj = super().__new__(cls, name, bases, dct)

        return cls_obj

    @classmethod
    def _construct_pathways(cls, pathways: Dict, dct: Dict) -> Dict:
        """
        Construct the pathways dictionary by examining decorated methods
        in the class dictionary.

        Parameters
        ----------
        pathways: Dict
            The initial pathways dictionary.
        dct: Dict
         The class attributes.

        Returns
        -------
        dict:
            The updated pathways dictionary.
        """
        for attr_name, attr_value in dct.items():
            # Skip attributes without `_solver_meta`
            if not hasattr(attr_value, "_solver_meta"):
                continue

            # Process each `_solver_meta` entry for the method
            for _meta in attr_value._solver_meta:
                # Validate that a path is specified
                _method_path = _meta.get("path")
                if not _method_path:
                    raise ValueError(
                        f"Method {attr_name} does not specify a path. "
                        "This is likely a bug in the developer's implementation."
                    )

                # Validate the type of the method
                _method_type = _meta.get("type")
                if _method_type not in ["process", "checker"]:
                    raise ValueError(
                        f"Method {attr_name} has invalid `type` in `_solver_meta`. "
                        "It must be either 'process' or 'checker'."
                    )

                # Ensure the pathway exists in the dictionary
                pathways.setdefault(
                    _method_path, {"processes": {}, "checkers": []}
                )

                if _method_type == "process":
                    # Process methods must define a step
                    _method_step = _meta.get("step")
                    if _method_step is None:
                        raise ValueError(
                            f"Method {attr_name} marked as 'process' is missing a 'step' in `_solver_meta`."
                        )
                    if _method_step in pathways[_method_path]["processes"]:
                        raise ValueError(
                            f"Duplicate step {_method_step} found in path '{_method_path}' for method {attr_name}."
                        )
                    pathways[_method_path]["processes"][_method_step] = {
                        'name': attr_name,
                        'args': _meta.get("args", None) or [],
                        'kwargs': _meta.get("kwargs", None) or {},
                        'desc': cls._process_attribute_description(attr_value),
                    }
                elif _method_type == "checker":
                    # Checker methods are added to the checkers list
                    pathways[_method_path]["checkers"].append(attr_name)

        return pathways

    @staticmethod
    def _process_attribute_description(attribute):
        # This method takes the attribute from the class dict and manipulates the __doc__ attribute to
        # get something we can use as a description for the process.
        doc = attribute.__doc__

        # Check if the documentation exists. If not, we just label with no description and
        # proceed.
        if doc is None:
            doc = "No Description."
            return doc

        # Check for leading \n --> this is common formatting to have issues with.
        if doc.startswith("\n"):
            doc = doc[2:]

        doc = doc.split("\n")[0]
        doc = doc.rstrip(" ").lstrip(" ")
        return doc


    @classmethod
    def _validate_pathways(cls, pathways: Dict):
        """
        Validate the constructed pathways to ensure correctness.

        Parameters
        ----------
        pathways: Dict
         The pathways dictionary.

        Raises
        ------
        ValueError:
            If any pathway is invalid.
        """
        for pathway, meta in pathways.items():
            # Ensure all process steps form a contiguous sequence
            process_steps = sorted(meta["processes"].keys())
            if process_steps != list(range(len(process_steps))):
                raise ValueError(
                    f"Pathway '{pathway}' has missing or non-contiguous steps: {process_steps}."
                )


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class Model(metaclass=_ModelMeta):
    r"""
    Base class for all astrophysical models in Pisces.

    The :py:class:`Model` class provides the foundation for implementing the astrophysical models
    which form the core of the Pisces code. While the particular details of a specific subclass may
    vary significantly, a :py:class:`Model` instances has a few core structures to be aware of:

    - **Grid Manager**:
      The grid manager (:py:attr:`~Model.grid_manager`) runs all of the HDF5 data operations and
      stores a set of ``fields`` containing different physical data. Generally, each physical property
      of the model is stored as a ``field`` in the grid manager. The grid manager then provides access
      to these fields and manages the creation / manipulation of new ones.

    - **Profiles**:
      In addition to the ``fields`` stored on the disk, :py:class:`Model` instances can also keep track of
      a set of analytic profiles (:py:class:`~pisces.profiles.base.Profile`) which pertain to its physics. Generally,
      this is used as a way to store a set of profiles which are used to build the model.

    - **Coordinate System**:
      Technically, the ``coordinate_system`` is attached to the :py:attr:`~Model.grid_manager`; however, its worth mentioning
      nonetheless. Some :py:class:`Model` sub-classes only allow certain coordinate systems.

    - **Solver**:
      An instance of :py:class:`~pisces.models.solver.ModelSolver` which is used to keep track of and execute the
      solution pipeline.

    By combining these 4 components, :py:class:`Model` allows for a very general set of models and construction procedures
    to be performed in subclasses.

    Attributes
    ----------
    ALLOWED_COORDINATE_SYSTEMS : Optional[List[str]]
        The coordinate systems which are compatible with this :py:class:`Model` class or subclass. The
        ``ALLOWED_COORDINATE_SYSTEMS`` class attribute should be specified as a list of ``str`` each of which
        corresponds to a class name of a particular coordinate system.
    DEFAULT_COORDINATE_SYSTEM : Optional[CoordinateSystem]
        The default coordinate system for the model. Used if no explicit coordinate
        system is provided during initialization.
    DEFAULT_COORDINATE_SYSTEM_PARAMS: Dict[str, Any]
        The default parameters to pass to the ``DEFAULT_COORDINATE_SYSTEM``.
    GRID_MANAGER_CLASS : Type[ModelGridManager]
        The grid manager class responsible for managing grid-related operations.
    INIT_FREE_AXES : List[str]
        The default free axes of the model. These are used when initializing the
        geometry handler for spatial operations.
    logger : LogDescriptor
        Specialized ``Logger`` instance used for handling logs within the model.

    See Also
    --------
    pisces.models.grids.base.ModelGridManager : Base class for grid managers.
    pisces.models.solver.ModelSolver : Handles pathways and solver execution.
    pisces.profiles.collections.HDF5ProfileRegistry : Manages profiles stored in HDF5 files.
    """
    # @@ VALIDATION MARKERS @@ #
    # These validation markers are used by the Model to constrain the valid
    # parameters for the model. Subclasses can modify the validation markers
    # to constrain coordinate system compatibility.
    ALLOWED_COORDINATE_SYSTEMS: Optional[List[str]] = None

    # @@ CLASS PARAMETERS @@ #
    DEFAULT_COORDINATE_SYSTEM: Optional[Type[CoordinateSystem]] = None
    DEFAULT_COORDINATE_SYSTEM_PARAMS: Dict[str, Any] = {}
    GRID_MANAGER_CLASS: Type[ModelGridManager] = ModelGridManager
    INIT_FREE_AXES: Optional[List[str]] = None
    logger: 'Logger' = LogDescriptor()

    # @@ LOADING METHODS @@ #
    # These methods form the core of the loading procedures for
    # models. Subclasses may overwrite these where necessary; however,
    # it is generally possible to leave the core of __init__ alone.
    def __init__(self, path: Union[str, Path]):
        r"""
        Initialize the Model instance by loading the HDF5 file and setting up managers.

        This method performs the following steps:
        1. Load the HDF5 file containing the model data.
        2. Initialize the grid manager to handle spatial grid operations.
        3. Validate the coordinate system to ensure compatibility with the model.
        4. Load profiles from the HDF5 file.
        5. Initialize the solver for executing computation pathways.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to the HDF5 file representing the model. If the path does not
            exist, an exception is raised.

        Raises
        ------
        ValueError
            If the specified path does not exist or is invalid.
        RuntimeError
            If the profile registry cannot be loaded.
        """
        # LOADING the handle from the path.
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Failed to find path: {self.path}")

        # LOADING the manager.
        self.logger.info("[LOAD] Loading model from file: %s.", self.path)

        # Load components
        self._manager = self.__class__.GRID_MANAGER_CLASS(self.path)
        self.logger.debug("[LOAD] (1/4) Grid manager loaded successfully.")

        self._validate_coordinate_system()
        self._geometry_handler = None
        self.logger.debug("[LOAD] (2/4) Coordinate system validated.")

        self._load_profiles()
        self.logger.debug("[LOAD] (3/4) Profiles loaded successfully.")

        # Set the solver
        self._solver = ModelSolver.from_hdf5(self)
        self.logger.debug("[LOAD] (4/4) Solver loaded successfully.")
        self.logger.info("[LOAD] COMPLETE: %s",self.path)

    def _load_profiles(self):
        """
        Load the profile registry from the HDF5 file.

        Raises
        ------
        RuntimeError
            If the profile registry cannot be loaded.
        """
        try:
            profiles_handle = self.handle.require_group("PROFILES")
            self._profiles = HDF5ProfileRegistry(profiles_handle)
        except Exception as e:
            raise RuntimeError("Failed to load profile registry.") from e

    # @@ DUNDER METHODS @@ #
    def __str__(self):
        return f"<{self.__class__.__name__}: path={self.path}>"

    def __repr__(self):
        return self.__str__()

    def __call__(self, overwrite: bool = False, pathway: Optional[str] = None):
        """
        Execute a pathway using the associated solver.

        This method delegates to the `ModelSolver` to execute a specific
        pathway on the model. If no pathway is specified, the solver's
        default pathway is used.

        Parameters
        ----------
        overwrite : bool, optional
            If True, allows execution even if the solver is already marked
            as solved. Defaults to False.
        pathway : Optional[str], optional
            The pathway to execute. If None, the default pathway is used.
            Defaults to None.

        Raises
        ------
        RuntimeError
            If the solver is already solved and `overwrite` is False.
        ValueError
            If the pathway is not valid or cannot be executed.
        """
        # VALIDATE that the solver exists if it couldn't be found from
        # the initialization process.
        if not hasattr(self, '_solver') or self._solver is None:
            raise RuntimeError("The solver is not initialized. Ensure the model has a valid solver.")

        pathway = pathway or self.path
        # RUNTIME
        self.logger.info("[EXEC] Solving model %s. ", self)
        self.logger.info("[EXEC] \tPIPELINE = %s. ", pathway)
        self.logger.info("[EXEC] \tNSTEPS = %s. ", len(self._pathways[pathway]['processes']))
        try:
            self._solver(pathway=pathway, overwrite=overwrite)
        except ValueError as e:
            self.logger.error("[EXEC] Failed to execute pathway '%s': %s", pathway, e)
            raise
        except RuntimeError as e:
            self.logger.error("[EXEC] Runtime error during execution of pathway '%s': %s", pathway, e)
            raise
        except Exception as e:
            self.logger.error("[EXEC] Error during execution of pathway '%s': %s", pathway, e)
            raise
        else:
            self.logger.info("[EXEC] Successfully executed pathway '%s'.", pathway or self._solver.default)

    # @@ VALIDATORS @@ #
    # These methods are for validating various types.
    @classmethod
    def _cls_validate_coordinate_system(cls,cs: 'CoordinateSystem'):
        """
        Validate the coordinate system of the model.

        Ensures the coordinate system matches allowed systems.

        Parameters
        ----------
        cs: CoordinateSystem
            The coordinate system to validate against.

        Raises
        ------
        ValueError
            If the coordinate system is not allowed.
        """
        if (cls.ALLOWED_COORDINATE_SYSTEMS is not None) and (
                cs.__class__.__name__ not in cls.ALLOWED_COORDINATE_SYSTEMS):
            raise ValueError(f"Invalid coordinate system for Model subclass {cls.__name__}: {cs}.\nThis error likely indicates"
                             f" that you are trying to initialize a structure with a coordinate system which is not compatible"
                             f" with the model.")

    def _validate_coordinate_system(self):
        """
        Validate the coordinate system of the model.

        Ensures the coordinate system matches allowed systems.

        Raises
        ------
        ValueError
            If the coordinate system is not allowed.
        """
        return self._cls_validate_coordinate_system(self.coordinate_system)

    # @@ CONSTRUCTION METHODS @@ #
    @classmethod
    def build_skeleton(cls,
                       path: Union[str, Path],
                       /,
                       bbox: ArrayLike,
                       grid_shape: ArrayLike,
                       chunk_shape: ArrayLike = None,
                       *,
                       overwrite: bool = False,
                       length_unit: str = 'kpc',
                       scale: Union[List[str], str] = 'linear',
                       profiles: Optional[Dict[str, 'Profile']] = None,
                       coordinate_system: Optional[CoordinateSystem] = None)-> HDF5_File_Handle:
        """
        Build the skeleton for initializing a new model.

        This method provides the basic HDF5 structure in ``path`` to initialize it as a
        :py:class:`Model` instance. This includes managing the profiles (``profiles``), the coordinate
        system (``coordinate_system``), and the grid shape / boundaries.

        Parameters
        ----------
        path : str or Path
            Path at which to create the model skeleton. If the path already exists,
            ``overwrite`` determines the behavior.
        bbox : ArrayLike
            Bounding box for the grid. This should be a ``(2,N)`` array or a type coercible to a ``(2,N)`` array. The first
            row corresponds to the lower-left corner of the grid and the second row corresponds to the
            upper-right corner of the grid.
        grid_shape : ArrayLike
            Shape of the grid. This should be an ``(N,)`` array of integers specifying the number of grid cells to place
            along each axis of the grid.
        chunk_shape : ArrayLike, optional
            Shape of chunks in the grid. The ``chunk_shape`` should follow the same conventions as ``grid_shape``;
            however, it must also be a whole factor of ``grid_shape`` (``grid_shape % chunk_shape == 0``). In some
            instances, operations may be performed in chunks instead of on the entire grid at once. In these cases,
            the chunk shape balances efficient computation with memory consumption.
        overwrite : bool, optional
            If True, overwrite any existing file at the specified path. If False and the
            file exists, an exception will be raised. Defaults to False.
        length_unit : str, optional
            The unit of measurement for grid lengths (e.g., ``'kpc'``, ``'Mpc'``). Defaults to ``'kpc'``.
        scale : list of str or str, optional
            The scaling type for each grid axis. Accepted values are ``'linear'`` or ``'log'``. If
            a single value is provided, it is applied to all axes. Defaults to ``'linear'``.
        profiles : dict[str, :py:class:`~pisces.profiles.base.Profile`], optional
            A dictionary of profiles to initialize in the model. Each key-value pair
            represents the profile name and corresponding :py:class:`~pisces.profiles.base.Profile` object.
        coordinate_system : CoordinateSystem, optional
            The coordinate system for the model. If None, the ``DEFAULT_COORDINATE_SYSTEM``
            of the class is used.

        Returns
        -------
        :py:class:`~pisces.io.hdf5.HDF5_File_Handle`
            A handle to the newly created HDF5 file representing the model.

        Raises
        ------
        ValueError
            If any of the input parameters fail validation, or the file path exists
            without ``overwrite=True``.

        Notes
        -----
        - The method validates the provided `coordinate_system` against the model's
          allowed systems.
        - The `bbox` and `grid_shape` parameters define the spatial extent and resolution
          of the model grid.

        See Also
        --------
        pisces.io.hdf5.HDF5_File_Handle : HDF5 file interface.
        pisces.models.grids.base.ModelGridManager : Manages grid structure and data.
        """
        cls.logger.info("[BLDR] Building model skeleton at path: %s", path)

        # VALIDATE path. Convert to a path object and manage overwriting.
        path = Path(path)
        if path.exists() and not overwrite:
            raise ValueError(f"The path '{path}' already exists. Use `overwrite=True` to overwrite.")
        elif path.exists() and overwrite:
            path.unlink()

        # VALIDATE coordinate system
        if coordinate_system is None:
            if cls.DEFAULT_COORDINATE_SYSTEM is not None:
                coordinate_system = cls.DEFAULT_COORDINATE_SYSTEM(**cls.DEFAULT_COORDINATE_SYSTEM_PARAMS)
            else:
                raise ValueError("Coordinate system must be provided to build the model skeleton.")

        # CREATE the file reference
        handle = HDF5_File_Handle(path, mode='w').switch_mode('r+')
        cls.logger.debug("[BLDR] HDF5 file created successfully.")

        # Initialize the grid manager skeleton
        ModelGridManager.build_skeleton(
            handle=handle,
            coordinate_system=coordinate_system,
            bbox=bbox,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
            length_unit=length_unit,
            scale=scale,
        )
        cls.logger.debug("[BLDR] Grid manager skeleton built successfully.")

        handle.require_group("PROFILES")
        if profiles:
            profile_registry = HDF5ProfileRegistry(handle["PROFILES"])
            for name, profile in profiles.items():
                profile_registry.add_profile(name, profile)
                cls.logger.debug("Profile '%s' added successfully.", name)

        cls.logger.info("[BLDR] Model skeleton created successfully at '%s'.", path)
        return handle

    # @@ UTILITY METHODS @@ #
    @classmethod
    def list_pipelines(cls)-> List[str]:
        """
        Return a list of the pipeline names present in the model class.

        Returns
        -------
        list of str
            A list of pipeline names.
        """
        return list(cls._pathways.keys())

    @classmethod
    def get_pipeline(cls,pipeline: str) -> Dict[str, Any]:
        """
        Fetch the data corresponding to a particular pipeline from the model class.

        Parameters
        ----------
        pipeline: str
            The name of the pipeline to fetch.

        Returns
        -------
        dict
            The pipeline data obtained.
        """
        try:
            return cls._pathways[pipeline]
        except KeyError:
            raise KeyError(f"The pipeline '{pipeline}' does not exist.")

    def summary(self) -> None:
        # Validate that we can get the tabulate package correctly. Otherwise we need
        # to raise an error.
        try:
            from tabulate import tabulate
        except ImportError:
            raise ValueError("Cannot import tabulate package.")

        # Build the tables using subfunctions and custom implementations.
        ptable, gtable, ftable = self.profiles.get_profile_summary(), self.grid_manager.get_grid_summary(), self.FIELDS.get_field_summary()

        model_table = tabulate([
            ['Path',str(self.path)],
            ['Solved',str(self._solver.is_solved)],
            ['Default Pathway',str(self._solver.default)],
        ], headers=['Attribute', 'Value'], tablefmt='grid')

        # Summary Tables
        print("\n===================== Model Summary =====================\n")

        print("General Information:")
        print("--------------------")
        print(model_table)

        print("\nAvailable Fields:")
        print("-----------------")
        print(ftable)

        print("\nAvailable Profiles:")
        print("-------------------")
        print(ptable)

        print("\nGrid Information:")
        print("-----------------")
        print(gtable)

        print("\n=========================================================\n")

    def pipeline_summary(self, pathway_name: str) -> None:
        """
        Generate and display a summary table for a specific solver pathway.

        This method provides details about the processes and checkers in the specified pathway,
        including step numbers, method names, arguments, keyword arguments, and descriptions.

        Parameters
        ----------
        pathway_name : str
            The name of the solver pathway to summarize.

        Raises
        ------
        ValueError
            If the specified pathway does not exist.
        ImportError
            If the `tabulate` package is not installed.

        Returns
        -------
        None
            Prints the pipeline summary to the console.

        """
        try:
            from tabulate import tabulate
        except ImportError:
            raise ImportError("The `tabulate` package is required to display the pipeline summary.")

        # Retrieve the pathway data
        if pathway_name not in self._pathways:
            raise ValueError(
                f"Pathway '{pathway_name}' does not exist. Available pathways: {list(self._pathways.keys())}")

        pathway = self._pathways[pathway_name]
        processes = pathway["processes"]
        checkers = pathway["checkers"]

        # Build table data for processes
        process_table = []
        for step, process in sorted(processes.items()):
            # Process the argument and kwarg strings. We want to use newlines to
            # ensure that we don't over-do the width.
            arg_string = "\n".join([
                arg if not isinstance(arg, list) else f"[\n{'\n'.join([str(i) for i in arg])}\n]"
                for arg in process["args"]
            ])
            kwarg_string = "{\n%s\n}"%("\n".join(["{k}={v}".format(k=k, v=v) for k, v in process['kwargs'].items()]))

            # Manage the process description.
            description_string = process['desc'] or 'No Description'

            if " " in description_string:
                ds = description_string.split(" ")
                ds_a,ds_b, ds_c = " ".join(ds[:len(ds)//3]), " ".join(ds[len(ds)//3:(2*len(ds))//3]), " ".join(ds[(2*len(ds))//3:])
                description_string = ds_a+"\n"+ds_b+"\n"+ds_c

            process_table.append([
                step,
                process['name'],
                arg_string,
                kwarg_string,
                description_string,
            ])

        # Build the process table
        process_headers = ["Step", "Process Name", "Arguments","Kwargs", "Description"]
        process_table_str = tabulate(process_table, headers=process_headers, tablefmt="grid")

        # Build the checkers list
        checkers_str = "\n".join(f"- {checker}" for checker in checkers) or "None"

        # Print the summary
        print(f"\n===================== Pipeline Summary: {pathway_name} =====================\n")

        print("Steps:")
        print("-------")
        print(process_table_str)

        print("\nCheckers:")
        print("---------")
        print(checkers_str)

        print("\n===============================================================================")

    # @@ PROPERTIES @@ #
    # These properties should generally not be altered as they
    # reference private attributes of the class; however, additional
    # properties may need to be implemented depending on the use case.
    @property
    def grid_manager(self) -> ModelGridManager:
        """
        Access the grid manager for the model.

        The grid manager is responsible for handling all grid-related operations,
        including spatial coordinates, field storage, and grid configuration.

        Returns
        -------
        :py:class:`pisces.models.grids.base.ModelGridManager`
            An instance of the grid manager for this model.

        Notes
        -----
        The grid manager interacts directly with the HDF5 file and handles grid
        initialization, chunking, and spatial alignment.
        """
        return self._manager

    # noinspection PyPep8Naming
    @property
    def FIELDS(self) -> 'ModelFieldContainer':
        """
        Access the fields stored in the model's grid.

        Fields are data arrays stored within the grid manager, typically representing
        physical quantities such as density, pressure, or temperature.

        Returns
        -------
        :py:class:`~pisces.models.grids.base.ModelFieldContainer`
            A container holding all fields managed
            by the grid manager.

        Notes
        -----
        The fields interface provides access to stored data arrays and allows retrieval
        or modification of grid-based quantities.
        """
        return self._manager.FIELDS

    @property
    def handle(self) -> HDF5_File_Handle:
        """
        Access the HDF5 file handle associated with the model.

        This handle provides a direct interface to the underlying HDF5 file, enabling
        interaction with raw data stored in the file.

        Returns
        -------
        :py:class:`~pisces.io.hdf5.HDF5_File_Handle`
            The HDF5 file handle for the model.

        Notes
        -----
        The handle is used internally for reading and writing data to the file.
        Direct interaction with the handle should be avoided unless necessary.
        """
        return self._manager.handle

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """
        Retrieve the coordinate system used by the model.

        The coordinate system defines the spatial framework (e.g., spherical,
        cylindrical, or Cartesian) over which the grid and profiles are defined.

        Returns
        -------
        :py:class:`~pisces.geometry.base.CoordinateSystem`
            The coordinate system object associated with the model.

        Notes
        -----
        The coordinate system must match one of the ``ALLOWED_COORDINATE_SYSTEMS``
        specified by the model. This ensures consistency between grid operations
        and the physics of the model.
        """
        return self._manager.coordinate_system

    @property
    def geometry_handler(self) -> GeometryHandler:
        """
        Retrieve the geometry handler for the model.

        The geometry handler manages operations that depend on the model's spatial
        framework, such as gradients, integrals, and field transformations.

        Returns
        -------
        :py:class:`~pisces.geometry.handler.GeometryHandler`
            An instance of the `GeometryHandler` class initialized with the model's
            coordinate system and free axes.

        Notes
        -----
        - If the geometry handler has not been initialized, it is created dynamically
          using the model's coordinate system and `INIT_FREE_AXES`.
        - The free axes are used to determine the directions along which operations
          such as gradients and integrals are performed.
        """
        if self._geometry_handler is None:
            if self.INIT_FREE_AXES is not None:
                free_axes = self.INIT_FREE_AXES
            else:
                free_axes = self.coordinate_system.AXES
            self._geometry_handler = GeometryHandler(self.coordinate_system, free_axes)
        return self._geometry_handler

    @property
    def profiles(self) -> HDF5ProfileRegistry:
        """
        Access the profiles stored in the model.

        Profiles are functional or tabulated representations of physical quantities
        (e.g., density, temperature) that can be evaluated over the grid.

        Returns
        -------
        :py:class:`~pisces.profiles.collections.HDF5ProfileRegistry`
            A registry containing all profiles associated with the model.

        Notes
        -----
        Profiles are typically used to initialize or compute grid fields. They can
        be stored in the HDF5 file and retrieved for further computations.
        """
        return self._profiles

    def add_field_from_profile(self,
                               profile_name: str,
                               *,
                               chunking: bool = False,
                               units: Optional[str] = None,
                               dtype: str = "f8",
                               overwrite: bool = False,
                               **kwargs):
        """
        Add a field to the grid by evaluating a stored profile.

        Parameters
        ----------
        profile_name : str
            The name of the profile stored in the model's `profiles` registry.
            The profile will be used to compute the field values.
        chunking : bool, optional
            If `True`, evaluate the profile in chunks to handle large grids. Default is `False`.
        units : Optional[str], optional
            The units of the field. If `None`, defaults to the units of the profile.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.
        **kwargs :
            Additional keyword arguments passed to the profile evaluation function.

        Raises
        ------
        ValueError
            If the profile name is not found in the model's profiles registry.
        """
        __logging__ = kwargs.pop('logging',True)
        # Retrieve the profile from the model's profile registry
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found in the model's profile registry.")

        profile = self.profiles[profile_name]

        self.grid_manager.add_field_from_profile(
            profile,
            profile_name,
            chunking=chunking,
            dtype=dtype,
            units=units,
            overwrite=overwrite,
            **kwargs
        )
        if __logging__:
            self.logger.info("[SLVR] Field '%s' added from internal profile.", profile_name)