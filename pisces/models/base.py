from pathlib import Path
from typing import Union, List, Dict, Optional, TYPE_CHECKING, Tuple
from numpy.typing import NDArray, ArrayLike
from pisces.models.grids.base import ModelGridManager
from pisces.models.solver import ModelSolver
from pisces.profiles.collections import HDF5ProfileRegistry
from pisces.io.hdf5 import HDF5_File_Handle
import numpy as np
from pisces.geometry.base import CoordinateSystem
from pisces.utilities.logging import LogDescriptor
import unyt
from collections import defaultdict

if TYPE_CHECKING:
    from pisces.profiles.base import Profile


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
                    }
                elif _method_type == "checker":
                    # Checker methods are added to the checkers list
                    pathways[_method_path]["checkers"].append(attr_name)

        return pathways

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

# noinspection PyAttributeOutsideInit
class Model(metaclass=_ModelMeta):
    """
    Base class for all astrophysical models in Pisces.

    Every :py:class:`~pisces.models.base.Model` represents a particular type
    of astrophysical system and self-consistently implements the relevant logic
    and physics.

    Attributes
    ----------
    ALLOWED_COORDINATE_SYSTEMS : Optional[List[str]]
        List of coordinate systems allowed for the model. Subclasses can override this.
        This ensures that only relevant / viable coordinate systems may be initialized as
        part of a given model.
    DEFAULT_COORDINATE_SYSTEMS : Optional[CoordinateSystem]
        Default coordinate system to use if none is specified during skeleton creation.
    logger : LogDescriptor
        Logger instance for the class.
    """
    # @@ VALIDATION MARKERS @@ #
    # These validation markers are used by the Model to constrain the valid
    # parameters for the model. Subclasses can modify the validation markers
    # to constrain coordinate system compatibility.
    ALLOWED_COORDINATE_SYSTEMS: Optional[List[str]] = None

    # @@ CLASS PARAMETERS @@ #
    DEFAULT_COORDINATE_SYSTEMS: Optional[CoordinateSystem] = None
    logger = LogDescriptor()

    def __init__(self, path: Union[str, Path]):
        """
        Initialize the Model instance by loading the HDF5 file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the HDF5 file representing the model.

        Raises
        ------
        ValueError
            If the specified path does not exist or is invalid.
        """
        # LOADING the handle from the path.
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Failed to find path: {self.path}")

        # LOADING the manager.
        self.logger.info("[LOAD] Loading model from file: %s.", self.path)

        # Load components
        self._manager = ModelGridManager(self.path)
        self.logger.debug("[LOAD] (1/4) Grid manager loaded successfully.")

        self._validate_coordinate_system()
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
        if not hasattr(self, '_solver') or self._solver is None:
            raise RuntimeError("The solver is not initialized. Ensure the model has a valid solver.")

        self.logger.info("[SLVR] Executing pathway '%s'.", pathway or self._solver.default)

        try:
            self._solver(pathway=pathway, overwrite=overwrite)
        except ValueError as e:
            self.logger.error("[SLVR] Failed to execute pathway '%s': %s", pathway, e)
            raise
        except RuntimeError as e:
            self.logger.error("[SLVR] Runtime error during execution of pathway '%s': %s", pathway, e)
            raise
        else:
            self.logger.info("[SLVR] Successfully executed pathway '%s'.", pathway or self._solver.default)

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
        Build the skeleton for a new Model.

        This method initializes the HDF5 structure, grid manager, and profiles.

        Parameters
        ----------
        path : Union[str, Path]
            Path at which to create the model skeleton. If the path already exists,
            ``overwrite`` determines the behavior.
        bbox : ArrayLike
            Bounding box for the grid.
        grid_shape : ArrayLike
            Shape of the grid.
        chunk_shape : ArrayLike, optional
            Shape of chunks in the grid.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to False.
        length_unit : str, optional
            Unit of length for the grid. Defaults to 'kpc'.
        scale : Union[List[str], str], optional
            Scaling type for each axis ('linear' or 'log'). Defaults to 'linear'.
        profiles : Optional[Dict[str, 'Profile']], optional
            Dictionary of profiles to initialize in the model.
        coordinate_system : Optional[CoordinateSystem], optional
            Coordinate system to use. Defaults to `DEFAULT_COORDINATE_SYSTEMS`.

        Raises
        ------
        ValueError
            If required parameters are missing or validation fails.
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
            if cls.DEFAULT_COORDINATE_SYSTEMS is not None:
                coordinate_system = cls.DEFAULT_COORDINATE_SYSTEMS
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

    @property
    def grid_manager(self) -> ModelGridManager:
        return self._manager

    # noinspection PyPep8Naming
    @property
    def FIELDS(self):
        return self._manager.FIELDS

    @property
    def handle(self):
        return self._manager.handle

    @property
    def coordinate_system(self):
        return self._manager.coordinate_system

    @property
    def profiles(self):
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
        **kwargs : dict
            Additional keyword arguments passed to the profile evaluation function.

        Raises
        ------
        ValueError
            If the profile name is not found in the model's profiles registry.
        """
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
        self.logger.info("[SLVR] Field '%s' added from internal profile.", profile_name)
