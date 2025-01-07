"""
Pisces astrophysics modeling base classes.
"""
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import unyt
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline

from pisces.geometry import GeometryHandler
from pisces.geometry.base import CoordinateSystem
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
from pisces.io.hdf5 import HDF5_File_Handle
from pisces.models.grids.base import ModelGridManager
from pisces.models.samplers import ModelSampler
from pisces.models.solver import ModelSolver
from pisces.models.utilities import ModelConfigurationDescriptor
from pisces.profiles.base import Profile
from pisces.profiles.collections import HDF5ProfileRegistry
from pisces.utilities.logging import LogDescriptor, devlog

if TYPE_CHECKING:
    from logging import Logger

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pisces.models.grids.base import ModelField, ModelFieldContainer
    from pisces.utilities.config import YAMLConfig


# noinspection PyProtectedMember
class ModelMeta(type):
    r"""
    Metaclass that inspects class attributes for solver-related metadata and constructs
    the internal ``_PATHWAYS`` dictionary on the resulting class. This dictionary is used
    by the solver to coordinate processes and checkers for each solver pathway.

    .. note::

       This metaclass is intended for use within the :py:class:`~pisces.models.base.Model`
       inheritance chain. It looks for methods marked with solver decorators (e.g.
       :py:func:`~pisces.models.solver.solver_process` and :py:func:`~pisces.models.solver.solver_checker`)
       and builds a structured description of each pathway, including:

       - A dictionary of process steps (indexed by step number).
       - A list of checker functions.

    **Key Steps**:

    1. **Collect** metadata from each decorated class method (methods that contain
       a :py:attr:`_solver_meta` attribute).
    2. **Partition** the methods by their declared pathway (``_solver_meta["path"]``).
    3. **Organize** them by type:
       - ``process`` entries (with a ``step``, ``args``, ``kwargs``, etc.)
       - ``checker`` entries (added to a list).
    4. **Validate** the integrity of these definitions (unique step numbers, valid types, etc.).
    5. **Store** the final pathways dictionary on the class as ``_PATHWAYS``.

    When a :py:class:`~pisces.models.solver.ModelSolver` (or similar) is later
    attached to the model, it can inspect ``_PATHWAYS`` to discover which processes
    and checkers to run in which order.
    """

    def __new__(
        mcs: Type["ModelMeta"],
        name: str,
        bases: Tuple[Type, ...],
        clsdict: Dict[str, Any],
    ) -> Type[Any]:
        r"""
        Create a new class object, searching for solver metadata in ``clsdict`` and
        building a ``_PATHWAYS`` attribute on the new class.

        Parameters
        ----------
        mcs : type
            The metaclass (this class).
        name : str
            The name of the class being created (e.g., "MyModel").
        bases : tuple
            The base classes for this new class.
        clsdict : dict
            The class attributes namespace (methods, class variables, etc.).

        Returns
        -------
        type
            The newly created class with the ``_PATHWAYS`` attribute populated or empty if the class is abstract.
        """
        # Identify if the class is abstract or concrete. This step is necessary so that
        # we don't force abstract classes to have properties which are not necessary given
        # that they are never instantiated.
        is_abstract = clsdict.get("_IS_ABC", False)

        # Create the base class object without adulteration from the
        # metaclass.
        cls = super().__new__(mcs, name, bases, clsdict)

        # If the class is not an abstract base class, we need to construct the pathways
        # dictionary. Otherwise, we leave the entire dictionary blank.
        cls._PATHWAYS = {}
        if not is_abstract:
            mcs.construct_pathways(cls, clsdict)

        return cls

    @staticmethod
    def construct_pathways(cls: "ModelMeta", clsdict: Dict[str, Any]) -> None:
        r"""
        Build the `_PATHWAYS` dictionary for a concrete (non-abstract) class by scanning all
        attributes (including inherited ones) for solver metadata.

        Parameters
        ----------
        cls :
            The class that is being constructed (i.e. the one with this metaclass).
        clsdict : dict
            The dictionary of attributes defined specifically on this class (not bases).

        Raises
        ------
        ValueError
            If any method is flagged incorrectly (e.g., missing a step for a process, or invalid type),
            or if there are duplicate steps within a pathway.
        """
        # We'll iterate over the class's entire MRO, collecting solver metadata from each base class as well.
        # This ensures that inherited methods with solver decorators are included.
        seen_attrs = set()
        inherits_pathways = clsdict.get("_INHERITS_PATHWAYS", False)
        if inherits_pathways:
            bases = cls.__mro__
        else:
            bases = [cls]

        for base in bases:
            # The built-in `object` or other sentinel classes won't contain relevant solver info.
            if base is object:
                continue

            # Explore the base's __dict__ to find candidate methods/attributes.
            for attr_name, method_obj in base.__dict__.items():
                # Make sure we don't re-process the same (base, attr_name) combination.
                if (base, attr_name) in seen_attrs:
                    continue
                seen_attrs.add((base, attr_name))

                # Now see if it has solver metadata.
                if not hasattr(method_obj, "_solver_meta"):
                    continue

                # For each solver meta record on that method, register it appropriately.
                for meta_record in method_obj._solver_meta:
                    path_name, entry_type = ModelMeta._validate_meta_entry(
                        attr_name, method_obj, meta_record
                    )

                    # Ensure that there's an entry for `path_name`.
                    if path_name not in cls._PATHWAYS:
                        cls._PATHWAYS[path_name] = {"processes": {}, "checkers": []}

                    if entry_type == "process":
                        ModelMeta._add_process_to_pathways(
                            cls=cls,
                            attribute_name=attr_name,
                            method=method_obj,
                            meta_record=meta_record,
                        )
                    elif entry_type == "checker":
                        cls._PATHWAYS[path_name]["checkers"].append(attr_name)

        # Finally, validate that the constructed `_PATHWAYS` is coherent, e.g. steps are contiguous.
        ModelMeta._validate_pathways(cls._PATHWAYS)

    @staticmethod
    def _add_process_to_pathways(
        cls: "ModelMeta", attribute_name: str, method: Any, meta_record: Dict[str, Any]
    ) -> None:
        r"""
        Insert a new "process" entry into the `_PATHWAYS` dictionary for a specific path.

        Parameters
        ----------
        cls :
            The class that is being constructed. Accesses `cls._PATHWAYS`.
        attribute_name : str
            The method name as it appears on the class.
        method : Any
            The actual method object (callable).
        meta_record : Dict[str, Any]
            The metadata dict. Must contain `path`, `step`, `args`, `kwargs`.

        Raises
        ------
        ValueError
            If no `step` is provided, or if `step` is already used.
        """
        path_name = meta_record["path"]
        step_no = meta_record.get("step")
        if step_no is None:
            raise ValueError(
                f"Method '{attribute_name}' is declared as a 'process' but has no 'step'. "
                f"Check your solver_process decorator usage."
            )

        # Ensure we haven't used this step in the same path already.
        if step_no in cls._PATHWAYS[path_name]["processes"]:
            raise ValueError(
                f"Duplicate step '{step_no}' found in path '{path_name}' for method '{attribute_name}'."
            )

        # Extract any user-specified arguments for the method call.
        process_args = meta_record.get("args", []) or []
        process_kwargs = meta_record.get("kwargs", {}) or {}

        # Prepare a short docstring as "desc".
        desc = ModelMeta._process_attribute_description(method)

        cls._PATHWAYS[path_name]["processes"][step_no] = {
            "name": attribute_name,
            "args": process_args,
            "kwargs": process_kwargs,
            "desc": desc,
        }

    @staticmethod
    def _validate_meta_entry(attribute_name: str, _: Any, meta_record: dict) -> tuple:
        r"""
        Validate a single metadata record to ensure it has the needed keys.

        Parameters
        ----------
        attribute_name : str
            The method name as it appears on the class.
        meta_record : dict
            The dictionary containing the solver metadata for this method.

        Returns
        -------
        tuple
            A 2-tuple of `(path_name, entry_type)`, where `entry_type` is "process" or "checker".

        Raises
        ------
        ValueError
            If there's no `path` in the record, or the `type` is invalid.
        """
        path_name = meta_record.get("path")
        entry_type = meta_record.get("type")

        if not path_name:
            raise ValueError(
                f"Method '{attribute_name}' is missing a 'path' in its solver metadata. "
                f"Likely a misconfigured or incomplete solver decorator."
            )
        if entry_type not in ("process", "checker"):
            raise ValueError(
                f"Method '{attribute_name}' includes an unsupported type '{entry_type}'. "
                f"Expected 'process' or 'checker'."
            )

        return path_name, entry_type

    @staticmethod
    def _process_attribute_description(method_obj: Any) -> str:
        r"""
        Extract a single-line docstring summary from the method object.

        Parameters
        ----------
        method_obj : Any
            Typically a callable that might have a `__doc__` attribute.

        Returns
        -------
        str
            The first line of the docstring, stripped of leading/trailing whitespace,
            or "No Description." if no docstring is present.
        """
        doc = getattr(method_obj, "__doc__", None)
        if doc is None:
            return "No Description."

        # Remove leading newlines/spaces; then take the first line.
        doc = doc.lstrip("\n").strip()
        first_line = doc.split("\n", 1)[0].strip()
        return first_line if first_line else "No Description."

    @staticmethod
    def _validate_pathways(pathways: Dict[str, Dict[str, Any]]) -> None:
        r"""
        Ensure all process steps in each pathway form a contiguous range [0..N-1].
        If not, raises a ValueError.

        Parameters
        ----------
        pathways : Dict[str, Dict[str, Any]]
            A dictionary of the form:
               {
                 "cooling_flow": {
                   "processes": { 0: {...}, 1: {...}, ... },
                   "checkers": [...]
                 },
                 ...
               }

        Raises
        ------
        ValueError
            If a pathway's steps are missing or non-contiguous.
        """
        for path_name, path_data in pathways.items():
            process_steps = sorted(path_data["processes"].keys())
            # E.g., if we have steps [0, 1, 3], we detect the gap at '2'.
            if process_steps != list(range(len(process_steps))):
                raise ValueError(
                    f"Pathway '{path_name}' has missing or non-contiguous steps: {process_steps}"
                )


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class Model(metaclass=ModelMeta):
    r"""
    Base class for physical models in Pisces.

    .. tip::

        For detailed information on basic use of the :py:class:`Model` class, see :ref:`modeling_overview`.

    The :py:class:`Model` provides a flexible framework for managing grid data, profiles, and solution workflows
    (pathways) for physical models (which are subclasses of this base class). At the core, the :py:class:`Model` has
    3 critical pieces of infrastructure:

    1. The **grid management** system (orchestrated by a :py:class:`~pisces.models.grids.base.ModelGridManager`) controls
       the physical domain of the model, the coordinate system for the model, and manages the storing / retrieving of fields
       containing the physical data of the model.

       You can access the grid manager via :py:attr:`grid_manager`, and the constituent fields which form the core
       of the model are accessed either via ``self.grid_manager.FIELDS` or (equivalently) :py:attr:`FIELDS`.

       .. note::

            In general, the :py:attr:`grid_manager` handles a lot of backend tasks for you. It's rarely necessary for a
            standard user to interact directly with it. For developers, it is where the core logic of the model is. Most importantly,
            the grid manager controls where and how data is stored, accessed, and manipulated in models.

    2. The **profiles** (:py:attr:`profiles`) are standard Pisces profiles (:py:class:`~pisces.profiles.base.Profile`) which
       are registered directly to the model. This allows for persistent access to them even when loading a model from disk.

    3. The **solver** system (orchestrated by a :py:class:`~pisces.models.solver.ModelSolver`) controls how the model is populated
       with data.

       .. tip::

            For standard users, this is the most important part of the whole system!

       Every :py:class:`Model` subclass has a set of so-called "pathways", which tell the class how to take an empty "skeleton" of
       the model to a fully realized, "solved," model. When you call the model (``model.solve_model()`` or ``model()``), the model
       instance attempts to execute one of these "pathways", proceeding in order through the procedures contained in the pathway.

       This allows for models to setup modular physics routines (to solve the Poisson Equation, manipulate equations of state, etc.)
       which can then be connected together to form a pathway which "solves" the model.

       .. note::

            Models may have several pathways! You can see the list of available pathways using the :py:meth:`list_pathways` method,
            and you can access details about the pathways using the :py:meth:`get_pathway` method. When you run your model, you
            can specify the ``pathway=`` argument to select a pathway to run.

       .. important::

            Pathways in models are not always accessible. They may rely on specific pieces of data which are present
            only in some cases or they may be specific to a certain case / parameter choice. Before running, every pathway
            calls a **validation** method to check if its permitted given the data available in the model.

    """
    # @@ VALIDATION MARKERS @@ #
    # These validation markers are used by the Model to constrain the valid
    # parameters for the model. Subclasses can modify the validation markers
    # to constrain coordinate system compatibility.
    #
    # : _IS_ABC : marks whether the model should seek out pathways or not.
    # : _INHERITS_PATHWAYS: will allow subclasses to inherit the pathways of their parent class.
    _IS_ABC: bool = True
    _INHERITS_PATHWAYS: bool = True

    # @@ CLASS PARAMETERS @@ #
    # The class parameters define several "standard" behaviors for the class.
    # These can be altered in subclasses to produce specific behaviors.
    DEFAULT_COORDINATE_SYSTEM: Optional[Type[CoordinateSystem]] = None
    """ :py:class:`~pisces.geometry.base.CoordinateSystem`: The default coordinate system.

    If a :py:class:`Model` skeleton is produced (either during ``__init__`` or during :py:meth:`Model.build_skeleton`) and
    a coordinate system is not provided, the default coordinate system will be used. The parameters (if any) used to
    generate the default coordinate system are provided by :py:attr:`Model.DEFAULT_COORDINATE_SYSTEM_PARAMS`.

    .. note::

        If :py:attr:`DEFAULT_COORDINATE_SYSTEM` is ``None``, then there is no default coordinate system and an
        error is raised if the user does not provide a coordinate system during the build process.

    """
    DEFAULT_COORDINATE_SYSTEM_PARAMS: Dict[str, Any] = {}
    """ dict of str, Any: The parameters for the default coordinate system.

    These parameters (if any) are passed to :py:attr:`Model.DEFAULT_COORDINATE_SYSTEM` during the build process if
    the default coordinate system is used.
    """
    INIT_FREE_AXES: Optional[List[str]] = None
    """ list of str: The model class's initial non-symmetric (free) axes.
    By specifying the :py:attr:`INIT_FREE_AXES` attribute, model developers may dictate the behavior of the
    :py:attr:`geometry_handler` attribute of any model instances.

    The idea behind this class-attribute is that specific models may have some "minimal symmetry" (i.e. galaxy cluster models
    which assume radial profile inputs). In these cases, even the blank :py:class:`Model` instance starts with some native
    symmetry mediated via the :py:attr:`geometry_handler` attribute.

    .. note::

        If ``INIT_FREE_AXES = None``, then it is assumed that all axes are free at baseline (there is no symmetry).

    .. tip::

        **For Developers**: Best practice regarding the :py:class:`Model` geometry handler depends largely on the desired
        behavior. In cases where pathways make frequent use of some "baseline symmetry", it may be worthwhile to implement this.
        If not, it can generally be left as ``None``.

    """
    GRID_MANAGER_TYPE: Type[ModelGridManager] = ModelGridManager
    """ Type[:py:class:`~pisces.models.grids.base.ModelGridManager`]: The grid manager type to use to manage this model class.

    The selected :py:class:`~pisces.models.grids.base.ModelGridManager` is the grid manager type that is used to manage this
    model class's grid structures. Implementing a custom grid manager class specific to particular model constraints is made
    possible by altering this attribute of the model class to point to the custom manager class.
    """
    SAMPLE_TYPE: Type["ModelSampler"] = ModelSampler
    """ Type[:py:class:`~pisces.models.samplers.ModelSampler`]: The sampler class to initialize when the
    model is asked to produce particles from its distributions.
    """
    # TODO: better doc.
    logger: "Logger" = LogDescriptor()
    """ Logger: The specialized logger for this model class.

    This specialized logger is controlled by the ``code``-type logger settings in the ``bin/config.yaml`` file. This allows
    the user to ensure that output from the model generation is turned off / on regardless of the behavior of the other
    logging systems in the Pisces ecosystem.
    """
    config: "YAMLConfig" = ModelConfigurationDescriptor()
    """ :py:class:`~pisces.utilities.config.YAMLConfig`: The configuration data for this model class.

    The configuration file provides access to things like field data, units, symbols, etc. for use throughout
    the model construction. If a configuration file is not specified, these utilities will not function when called.
    """

    # @@ LOADING METHODS @@ #
    # These methods form the core of the loading procedures for
    # models. Subclasses may overwrite these where necessary; however,
    # it is generally possible to leave the core of __init__ alone.
    def __init__(self, path: Union[str, Path]):
        r"""
        Initialize the :py:class:`Model` instance by loading from an HDF5 file.

        .. tip::

            Using ``__init__`` to load a :py:class:`Model` from disk is generally the best approach
            when you have an existing model. Otherwise, you may want to use a class method which directs
            the solution process to occur on creation.

        Parameters
        ----------
        path : str or Path
            The file path to the HDF5 file representing the model. If the path does not
            exist, an exception is raised.

            .. note::

                The specified path must point to a valid :py:class:`Model` instance on disk. If it does not,
                any number of exceptions may be raised due to incorrect data types and other issues.

        Raises
        ------
        ValueError
            If the specified path does not exist or is invalid.
        RuntimeError
            If the profile registry cannot be loaded.

        Notes
        -----

        **Initialization Procedure**:

        Generally, the initialization procedure occurs in the following order:

        1. Load the HDF5 file containing the model data.
        2. Initialize the grid manager to handle spatial grid operations.
        3. Validate the coordinate system to ensure compatibility with the model.
        4. Load profiles from the HDF5 file.
        5. Initialize the solver for executing computation pathways.

        """
        # Pull the user's provided path and ensure that it exists.
        # We do not permit non-existent paths to proceed.
        self._path: Path = Path(path)  # -> private to use property instead.
        if not self.path.exists():
            raise ValueError(f"Failed to find path: {self.path}")
        self.logger.info("[LOAD] Loading model from file: %s...", self._path)

        # Load the components of the Model from disk.
        # There are 3 constituent attributes to generate:
        #   1. The grid manager.
        #      This includes an additional validation step.
        #   2. Profiles.
        #   3. The solver.
        #
        # => Start with the grid manager / coordinate system.
        self._geometry_handler = None  # --> holds the lazy loaded geometry handler.
        self._manager = self.__class__.GRID_MANAGER_TYPE(self._path)
        self._validate_coordinate_system()
        devlog.debug("(model load) got manager: %s.", self._manager)
        devlog.debug(
            "(model load) got coordinate_system: %s.", self._manager.coordinate_system
        )

        # => Load the profiles into the collection.
        self._load_profiles()
        devlog.debug("(model load) got profiles: %s", self._profiles.keys())

        # => Load the solver.
        # This is always a simple ModelSolver (the solver class is really simple).
        self._solver = ModelSolver.from_hdf5(self)
        devlog.debug("(model load) got solver: %s", self._solver)

        # Log output to standard log so users can see that the
        # load completed.
        self.logger.info("[LOAD] COMPLETE: %s", self._path)

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
    def build_skeleton(
        cls,
        path: Union[str, Path],
        /,
        bbox: ArrayLike,
        grid_shape: ArrayLike,
        chunk_shape: ArrayLike = None,
        *,
        overwrite: bool = False,
        length_unit: str = None,
        scale: Union[List[str], str] = None,
        profiles: Optional[Dict[str, "Profile"]] = None,
        coordinate_system: Optional[CoordinateSystem] = None,
    ) -> HDF5_File_Handle:
        """
        Construct a "skeleton" for the :py:class:`Model` class.

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
        bbox : NDArray[np.float64], optional
            The bounding box that defines the physical extent of the grid in each axis. This should be convertible into
            a :py:class:`~pisces.models.grids.structs.BoundingBox`, with shape ``(2, NDIM)``, where ``NDIM`` matches
            the number of axes in ``coordinate_system``. The first row contains the minimum coordinate
            values along each axis, and the second row contains the maximum coordinate values.

            .. note::

               You can provide this in a Python list form such as ``[[x0_min, x0_max],
               [x1_min, x1_max], ...]``.

        grid_shape : NDArray[np.int64], optional
            The shape of the grid, specifying the number of cells along each axis. It should be a 1D array-like of
            integers with length equal to the number of dimensions in
            ``coordinate_system``.

        chunk_shape : NDArray[np.int64], optional
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
            raise ValueError(
                f"The path '{path}' already exists. Use `overwrite=True` to overwrite."
            )
        elif path.exists() and overwrite:
            path.unlink()

        # VALIDATE coordinate system
        if coordinate_system is None:
            if cls.DEFAULT_COORDINATE_SYSTEM is not None:
                coordinate_system = cls.DEFAULT_COORDINATE_SYSTEM(
                    **cls.DEFAULT_COORDINATE_SYSTEM_PARAMS
                )
            else:
                raise ValueError(
                    "Coordinate system must be provided to build the model skeleton."
                )

        # CREATE the file reference
        handle = HDF5_File_Handle(path, mode="w").switch_mode("r+")
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

    # @@ VALIDATORS @@ #
    # These methods are for validating various types.
    @classmethod
    def _cls_validate_coordinate_system(cls, cs: "CoordinateSystem"):
        r"""
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
        if (cls.GRID_MANAGER_TYPE.ALLOWED_COORDINATE_SYSTEMS is not None) and (
            cs.__class__.__name__
            not in cls.GRID_MANAGER_TYPE.ALLOWED_COORDINATE_SYSTEMS
        ):
            raise ValueError(
                f"Invalid coordinate system for Model subclass {cls.__name__}: {cs}.\nThis error likely indicates"
                f" that you are trying to initialize a structure with a coordinate system which is not compatible"
                f" with the model."
            )

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

    # @@ DUNDER METHODS @@ #
    def __str__(self):
        return f"<{self.__class__.__name__}: path={self._path}>"

    def __repr__(self):
        return self.__str__()

    def __call__(self, overwrite: bool = False, pathway: Optional[str] = None):
        self.solve_model(overwrite=overwrite, pathway=pathway)

    def __getitem__(self, item: str) -> "ModelField":
        try:
            return self.profiles[item]
        except KeyError:
            raise KeyError(f"Model {self} has no field named {item}.")

    def __setitem__(self, _: str, __: Any):
        raise NotImplementedError(
            "Model objects do not permit direct setting using __setitem__."
        )

    def __len__(self) -> int:
        return len(self.FIELDS)

    def __eq__(self, other: "Model") -> bool:
        return self.path == other.path

    def __iter__(self):
        return self.FIELDS.__iter__()

    def __delitem__(self, key: str):
        if key in self.FIELDS:
            self.FIELDS.__delitem__(key)
        else:
            raise KeyError(f"Model {self} has no field named {key}.")

    def __contains__(self, item: str) -> bool:
        return str in self.FIELDS

    # @@ PROPERTIES @@ #
    # These properties should generally not be altered as they
    # reference private attributes of the class; however, additional
    # properties may need to be implemented depending on the use case.
    @property
    def path(self) -> Path:
        """
        The file path associated with this :py:class:`Model` instance's location on disk.

        Returns
        -------
        Path
        """
        return self._path

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
    def grid_manager(self) -> ModelGridManager:
        r"""
        The grid manager for the model.

        The grid manager (:py:class:`~pisces.models.grids.base.ModelGridManager`) is responsible for handling all grid-related operations,
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
    def FIELDS(self) -> "ModelFieldContainer":
        r"""
        Access the fields stored in the model's :py:attr:`grid_manager`.

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
    def coordinate_system(self) -> CoordinateSystem:
        r"""
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
        r"""
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
        r"""
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

    @property
    def particle_sampler(self) -> ModelSampler:
        r"""
        The model sampler attached to this model.
        """
        # TODO: better doc.
        if not hasattr(self, "_particle_sampler"):
            self._particle_sampler = ModelSampler(self)

        return self._particle_sampler

    @property
    def is_solved(self) -> bool:
        """
        Boolean flag indicating whether the model is solved.

        Returns
        -------
        bool
        """
        return self._solver.is_solved

    @property
    def default_pathway(self) -> Optional[str]:
        """
        The default pathway associated with this model instance.

        Returns
        -------
        str
        """
        return self._solver.default

    @default_pathway.setter
    def default_pathway(self, pathway: str):
        if pathway in self._solver.list_pathways():
            self._solver.default = pathway
        else:
            raise ValueError(
                f"Pathway {pathway} is not recognized for models of class {self.__class__.__name__}.\n"
                f"Recognized pathways are {list(self._solver.list_pathways())}."
            )

    # @@ UTILITY FUNCTIONS @@ #
    # These utility functions are used throughout the model generation process for various things
    # and are not sufficiently general to be worth implementing elsewhere.
    def _assign_default_units_and_add_field(
        self,
        field_name: str,
        field_data: unyt.unyt_array,
        create_field: bool = True,
        axes: Optional[List[str]] = None,
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

        create_field : bool, optional
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
        # Validate the input axes. By default, we'll just adopt the standard initial axes and / or
        # rely on the full axes set from the coordinate system.
        if (axes is None) and (self.INIT_FREE_AXES is not None):
            axes = self.INIT_FREE_AXES
        elif axes is None:
            axes = self.coordinate_system.AXES
        else:
            pass

        # Attempt to look up the default units from the configuration file. If the configuration
        # doesn't exist, this will cause an error to be raised.
        try:
            units = self.get_default_units(field_name)
        except KeyError as e:
            raise ValueError(
                f"No default units defined for field '{field_name}'."
            ) from e

        field_data = field_data.to(units)

        # Manage the field creation process. If we are adding the field, it can just be passed off
        # to the field manager as normal. The logging statement is standardized across all
        # models inheriting this method.
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
                field_data.ndim,
            )

        # Return the field so that it can be accessed if this was just a unit validation method.
        return field_data

    def _validate_field_dependencies(self, field_name: str, required_fields: dict):
        """
        Validate that all required fields exist for the equation of state computation.
        """
        missing = [f for f in required_fields[field_name] if f not in self.FIELDS]
        if missing:
            raise ValueError(f"Cannot compute {field_name}. Missing fields: {missing}")

    def _coerce_extent_for_plots(
        self, extent: Any
    ) -> Tuple[unyt.unyt_array, np.ndarray]:
        """
        Basic utility function to coerce extent args passed to plotting
        routines.

        Parameters
        ----------
        extent: Any
        The extent to coerce.
        """
        if isinstance(extent, tuple):
            # Extent provided as (array, unit) tuple.
            try:
                extent = unyt.unyt_array(extent[0], extent[1])
            except Exception as e:
                raise ValueError(f"Invalid extent parameter: {e}.") from e
        elif not isinstance(extent, unyt.unyt_array):
            # Attempt to convert to unyt_array using the built-in length units.
            try:
                extent = unyt.unyt_array(extent, self.grid_manager.length_unit)
            except Exception as e:
                raise ValueError(f"Invalid extent parameter: {e}.") from e
        else:
            # naturally an unyt array.
            pass

        extent_scaled = extent.to_value(self.grid_manager.length_unit)
        extent = extent

        return extent, extent_scaled

    # @@ CORE METHODS @@ #
    # These are the standard methods of the class. They
    # should generally not need to be changed. Any subclass-specific
    # capabilities are reserved for implementation in the
    # relevant subclass.
    @classmethod
    def get_pathway(cls, pathway: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Fetch information about a particular solution pathway in this :py:class:`Model` class.

        Each pathway in the model has 2 components:

        1. **Processes** (``dict``): The list of steps in the pathway. Each entry in the processes dictionary
           is an ``int`` corresponding to the order of the process in the pathway. Each value in the processes
           dictionary is another dictionary with the following information:

           - ``name`` (``str``): The name of the method (of :py:class:`Model` / subclass thereof) which executes this
             step in the pathway.
           - ``args`` (``list[Any]``): Additional arguments (aside from the :py:class:`Model` instance) which are
             passed to this step in the pathway.
           - ``kwargs`` (``dict[Any, Any]``): Additional keyword arguments which are passed to this step in the pathway.
           - ``desc`` (``str``): The short description of the step process.

        2. **Checkers** (``list[str]``): The list of checker methods (of :py:class:`Model` / subclass thereof) which are
           called to validate this pathway.

        Parameters
        ----------
        pathway: str
            The name of the pathway to fetch.

        Returns
        -------
        dict
            The processes dictionary.
        list
            The checker methods list.
        """
        try:
            return (
                cls._PATHWAYS[pathway]["processes"],
                cls._PATHWAYS[pathway]["checkers"],
            )
        except KeyError:
            raise KeyError(f"The pathway '{pathway}' does not exist.")

    @classmethod
    def get_pathway_step(cls, pathway: str, step: int):
        """
        Fetch information about a particular step in a solution pathway of this :py:class:`Model` class.

        Each step in the pathway has the following components:

        - ``name`` (``str``): The name of the method (of :py:class:`Model` / subclass thereof) which executes this
          step in the pathway.
        - ``args`` (``list[Any]``): Additional arguments (aside from the :py:class:`Model` instance) which are
          passed to this step in the pathway.
        - ``kwargs`` (``dict[Any, Any]``): Additional keyword arguments which are passed to this step in the pathway.
        - ``desc`` (``str``): The short description of the step process.

        Parameters
        ----------
        pathway: str
            The name of the pathway to fetch.
        step: int
            The index of the step in the pathway.

        Returns
        -------
        dict
        """
        try:
            return cls.get_pathway(pathway)[0][step]
        except IndexError:
            raise ValueError(
                f"The pathway '{pathway}' does not have the step '{step}'."
            )
        except KeyError as e:
            raise e

    @classmethod
    def get_pathway_length(cls, pathway: str) -> int:
        """
        Return the length of a specific pathway in the model.

        Parameters
        ----------
        pathway: str

        Returns
        -------
        int
        """
        processes, checkers = cls.get_pathway(pathway)
        return len(processes)

    @classmethod
    def list_pathways(cls) -> List[str]:
        """
        Return a list of the pathway names present in this :py:class:`Model` class.

        Returns
        -------
        list of str
            A list of pathways names.
        """
        return list(cls._PATHWAYS.keys())

    @classmethod
    def pathway_summary(cls, pathway_name: str) -> None:
        r"""
        Display a tabulated summary of the specified solver pathway for this model class.

        This class method retrieves details about a given pathway from the class-level
        :py:attr:`_PATHWAYS` dictionary, builds a table of process steps (including step
        numbers, arguments, keyword arguments, and short descriptions), and prints that table
        alongside a list of checkers for quick reference.

        Parameters
        ----------
        pathway_name : str
            The name of the solver pathway to summarize. Must match a key in
            :py:attr:`_PATHWAYS`.

        Raises
        ------
        ImportError
            If the ``tabulate`` package is not installed, which is required to format the table.
        ValueError
            If the specified ``pathway_name`` does not exist in :py:attr:`_PATHWAYS`.

        Examples
        --------
        .. code-block:: python

            # Suppose your Model class or subclass has a pathway named "cooling_flow".
            # You can display its summary (steps, arguments, checkers) like so:

            MyModel.pathway_summary("cooling_flow")

        Notes
        -----
        - The process steps are sorted by their step number in ascending order.
        - Arguments and keyword arguments (``args`` / ``kwargs``) are displayed in
          multi-line format for readability.
        - If no checkers exist for this pathway, "None" is displayed.
        - The first line of each process method's docstring is shown in the "Description"
          column, if available.
        """
        try:
            from tabulate import tabulate
        except ImportError as exc:
            raise ImportError(
                "The 'tabulate' package is required to display a pathway summary. "
                "Please install it via 'pip install tabulate'."
            ) from exc

        # Ensure the pathway exists in this class's _PATHWAYS dictionary.
        if pathway_name not in cls._PATHWAYS:
            valid = list(cls._PATHWAYS.keys())
            raise ValueError(
                f"Pathway '{pathway_name}' does not exist. "
                f"Available pathways: {valid}"
            )

        # Retrieve relevant data about the pathway.
        pathway_data = cls._PATHWAYS[pathway_name]
        processes = pathway_data["processes"]
        checkers = pathway_data["checkers"]

        # Prepare a table of processes, sorted by step number.
        process_table = []
        for step_number, process_info in sorted(processes.items(), key=lambda x: x[0]):
            # Format arguments in a multi-line string.
            arg_lines = []
            for arg_item in process_info["args"]:
                if isinstance(arg_item, list):
                    # For lists, we show them in a bracketed, multi-line fashion.
                    contents = "\n".join(str(subitem) for subitem in arg_item)
                    arg_lines.append(f"[\n{contents}\n]")
                else:
                    arg_lines.append(str(arg_item))
            arg_string = "\n".join(arg_lines)

            # Format keyword arguments similarly (multi-line).
            kwarg_lines = []
            for k, v in process_info["kwargs"].items():
                kwarg_lines.append(f"{k}={v}")
            kwarg_string = (
                "{\n" + ("\n".join(kwarg_lines)) + "\n}" if kwarg_lines else "{}"
            )

            # Use the short docstring snippet if available; fallback if empty or None.
            desc = process_info.get("desc", "No Description") or "No Description"

            process_table.append(
                [
                    step_number,
                    process_info["name"],
                    arg_string,
                    kwarg_string,
                    desc,
                ]
            )

        # Column headers for the table of processes.
        headers = [
            "Step",
            "Process Name",
            "Arguments",
            "Keyword Arguments",
            "Description",
        ]
        process_table_str = tabulate(process_table, headers=headers, tablefmt="grid")

        # Format the checker list (if any).
        if checkers:
            checkers_str = "\n".join(f"- {checker}" for checker in checkers)
        else:
            checkers_str = "None"

        # Print the assembled summary.
        print(
            f"\n====================== Pathway Summary: '{pathway_name}' ======================\n"
        )
        print("Steps:")
        print("------")
        print(process_table_str)
        print("\nCheckers:")
        print("---------")
        print(checkers_str)
        print(
            "\n==============================================================================="
        )

    @classmethod
    def get_default_units(cls, field_name: str) -> unyt.Unit:
        """
        Retrieve the default units for a specified field.

        This method looks up the default units for a given field name from the
        :py:attr:`~pisces.models.base.Model.config` configuration
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
            If the specified field name is not found in the :py:attr:`~pisces.models.base.Model.config`
            dictionary.

        """
        # Fetch the unit data from the field entry in the
        # parameters object.
        try:
            _unit_str = cls.config[f"fields.{field_name}.units"]
            if _unit_str is None:
                raise KeyError()
        except KeyError:
            raise ValueError(
                "Failed to get default units for field `%s` because it is not a known field."
                % field_name
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                f"Failed to access the configuration utility for model class {cls.__name__}: {e}"
            )

        # Attempt to return the unit
        try:
            return unyt.Unit(_unit_str)
        except Exception as e:
            raise ValueError(
                f"Failed to convert string unit {_unit_str} to unyt.Unit object: {e}"
            )

    def solve_model(self, overwrite: bool = False, pathway: str = None):
        """
        Solve this model using one of the solution pathways associated with it.

        Every physical model in Pisces implements one (or many) solution "pathways" which take
        the model from an "empty" state to a completed state. This process is called "solving" the model
        and is achieved by calling the :py:meth:`solve_model` method on the model in question.

        Parameters
        ----------
        overwrite : bool, optional
            If ``True``, then the :py:class:`Model` will be solved even if it has already been solved.
            This will overwrite any data that was written during a previous solution if necessary.

            .. warning::

                This can be disastrous if done unintentionally! Make sure you do not set ``overwrite=True`` lightly.
                Solving models can be expensive and time-consuming. Loosing all your data to have to solve the model again
                is both inefficient and potentially frustrating.

        pathway : Optional[str], optional

            The solution pathway to execute. If ``pathway`` is not specified, then the solver will attempt to use
            the default pathway (:py:attr:`default_pathway`). If no default pathway exists, an error is raised insisting
            that a pathway choice be provided.

        Raises
        ------
        RuntimeError
            If the solver is already solved and ``overwrite`` is False.
        ValueError
            If the pathway is not valid or cannot be executed.
        """
        # Validate that the solver exists for this class and set the pathway / default pathway.
        # produce some logging information for the user.
        if not hasattr(self, "_solver") or self._solver is None:
            raise RuntimeError(
                "The solver is not initialized. This error shouldn't be possible..."
            )

        pathway = pathway or self.default_pathway
        if pathway is None:
            raise ValueError(
                f"No `pathway` was provided and {self} has no default pathway set.\n"
                f"Please specify a pathway or a default pathway to solve the model."
            )

        # Produce the basic logging information for the user.
        self.logger.info("[EXEC] Solving model %s. ", self)
        self.logger.info("[EXEC] \tPATHWAY = %s. ", pathway)
        self.logger.info("[EXEC] \tNUM_STEPS = %s. ", self.get_pathway_length(pathway))

        # Pass the runtime off to the solver to continue the solution procedure.
        try:
            self._solver(pathway=pathway, overwrite=overwrite)
        except RuntimeError as e:
            self.logger.error(
                "[EXEC] Runtime error during execution of pathway '%s': %s",
                pathway,
                e.__str__(),
            )
            raise
        except Exception as e:
            self.logger.error(
                "[EXEC] Error during execution of pathway '%s': %s",
                pathway,
                e.__str__(),
            )
            raise
        else:
            self.logger.info(
                "[EXEC] Successfully executed pathway '%s'.",
                pathway or self._solver.default,
            )

    def summary(self) -> None:
        r"""
        Display a comprehensive console summary of the current model state.

        This method invokes the ``tabulate`` library (if available) to present various
        aspects of the model in a tabular format, such as:

        1. **General Information**:
           - The model's file ``path`` (if any).
           - Whether the model is already solved.
           - The default pathway (if one is set).

        2. **Available Fields**:
           A table enumerating all fields in the model, retrieved from
           :py:meth:`self.FIELDS.get_field_summary`, showing information
           such as field names, units, shapes, and axes.

        3. **Available Profiles**:
           A summary of the model's profiles, as returned by
           :py:meth:`self.profiles.get_profile_summary`, typically including
           profile names and relevant metadata.

        4. **Grid Information**:
           Details about the grid structure (bounding box, chunk shape, scale,
           etc.), via :py:meth:`self.grid_manager.get_grid_summary`.

        The summary is printed directly to stdout. Use this for a quick
        diagnostic or overview of the model's configuration and data.

        Raises
        ------
        ValueError
            If the ``tabulate`` library cannot be imported (i.e., it is not installed).

        Examples
        --------
        .. code-block:: python

            my_model = SomeDerivedModel(...)
            # ... possibly load or compute fields ...
            my_model.summary()

            # Outputs a multi-section summary of the model's current state, including
            # any fields, profiles, and grid setup.

        Notes
        -----
        - If you wish to display more granular details about a specific solver
          pathway, see :py:meth:`Model.pathway_summary` (class-level) or
          :py:meth:`pathway_summary` (instance-level method, if defined).
        - The layout of the printed summary may be subject to terminal width
          or formatting constraints.
        """
        # Validate that we can get the tabulate package correctly. Otherwise we need
        # to raise an error.
        try:
            from tabulate import tabulate
        except ImportError:
            raise ValueError("Cannot import tabulate package.")

        # Build the tables using subfunctions and custom implementations.
        ptable, gtable, ftable = (
            self.profiles.get_profile_summary(),
            self.grid_manager.get_grid_summary(),
            self.FIELDS.get_field_summary(),
        )

        model_table = tabulate(
            [
                ["Path", str(self.path)],
                ["Solved", str(self.is_solved)],
                ["Default Pathway", str(self.default_pathway)],
            ],
            headers=["Attribute", "Value"],
            tablefmt="grid",
        )

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

    def plot_slice(
        self,
        field_name: str,
        extent: Any,
        view_axis: Optional[Literal["x", "y", "z"]] = "z",
        figure: Optional["Figure"] = None,
        axes: Optional["Axes"] = None,
        resolution: Optional[Tuple[float, float]] = (500, 500),
        view_axis_position: Optional[Any] = 0,
        **kwargs,
    ):
        from pisces.utilities.plotting import construct_subplot

        # Setup the subplot / figure to ensure that they
        # are present
        figure, axes = construct_subplot(figure, axes)

        # Validate the extent attribute. May be an unyt_array or Tuple(list, str) or simply
        # a basic list-like object. All cases need to be handled naturally.
        extent_image, extent_backend = self._coerce_extent_for_plots(extent)

        # Construct the image array from inputs.
        image_array = self.grid_manager.generate_slice_image_array(
            field_name,
            str(view_axis),
            extent_backend,
            np.array(resolution, dtype=np.uint16),
            view_axis_position,
        )

        # Check for non-plotting calls. This allows users more control over
        # plot routines by simply getting the image out on its own.
        if kwargs.pop("noplot", False):
            return image_array, figure, axes

        # Add the image to the plot. Utilize the normalization passed via kwargs
        # if necessary and the colormap.
        imshow_object = axes.imshow(image_array.T, extent=extent_image.d, **kwargs)

        # Managing axes labels, ticks, etc.
        _axes_labels = [ax for ax in ["x", "y", "z"] if ax != view_axis]
        axes.set_xlabel(
            r"$%s$ / $\left[%s\right]$"
            % (_axes_labels[0], extent_image.units.latex_repr)
        )
        axes.set_ylabel(
            r"$%s$ / $\left[%s\right]$"
            % (_axes_labels[1], extent_image.units.latex_repr)
        )

        field_label = self.config[f"fields.{field_name}.label"]
        if field_label is None:
            field_label = field_name

        figure.colorbar(
            imshow_object,
            ax=axes,
            label=r"%s / $\left[%s\right]$"
            % (field_label, self.FIELDS[field_name].units.latex_repr),
        )

        return image_array, figure, axes

    # noinspection PyIncorrectDocstring
    def add_field_from_function(self, function: Callable, field_name: str, **kwargs):
        """
        Create a :py:class:`~pisces.models.grids.base.ModelField` in this model's
        :py:class:`~pisces.models.grids.base.ModelGridManager` by evaluating the provided function.

        This method takes a function (``function``) and evaluates it at the relevant grid points to generate
        a new field with name ``field_name``.

        Parameters
        ----------
        function : Callable
            A function which takes (as input) ``N`` arguments ``(x_1,...,x_N)`` corresponding to the coordinate
            values of the ``N`` axes specified by the ``axes`` argument. If ``axes`` is not specified, then ``N=NDIM``, where
            ``NDIM`` is the number of dimensions in the coordinate system.
        field_name : str
            The name to give to the newly generated field.

            .. note::

                The ``field_name`` will be the location in the HDF5 file as well (``FIELDS/field_name``).

        axes : Optional[List[str]], optional
            The coordinate axes along which the function is to be evaluated. If ``axes`` is not provided, then
            it is assumed that the function operates on all the coordinates of the coordinate system.

            .. hint::

                Ensure that ``axes`` is self-consistent with the call signature of the ``function`` parameter.

        chunking : bool, optional
            If `True`, evaluate the function in chunks. Default is `False`.

            .. tip::

                This is generally not necessary unless you cannot load the entire base grid into memory at once. This
                is particularly common if the function is operating in 3 or more dimensions, in which case even moderately
                resolved grids may take up significant memory.

        units : Optional[str], optional
            The units to give to the field. If ``units`` is not provided, then it is assumed that the field is
            dimensionless.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.

        Raises
        ------
        ValueError
            If the function, axes, or other parameters are invalid.
        """
        __logging__ = kwargs.pop("logging", True)

        # Pass to the grid manager to add the field.
        self.grid_manager.add_field_from_function(function, field_name, **kwargs)

        # log the output.
        if __logging__:
            self.logger.debug("[SLVR] Field '%s' added from function.", field_name)

    # noinspection PyIncorrectDocstring
    def add_field_from_profile(
        self, profile: Union[str, Profile], field_name: Optional[str] = None, **kwargs
    ):
        """
        Create a :py:class:`~pisces.models.grids.base.ModelField` in this model's
        :py:class:`~pisces.models.grids.base.ModelGridManager` by evaluating a profile.

        This method effectively wraps the :py:meth:`ModelGridManager.add_field_from_function` but utilizes the axes
        information from the profile to reduce the number of necessary inputs.

        Parameters
        ----------
        profile : str or :py:class:`~pisces.profiles.base.Profile`
            Any valid :py:class:`~pisces.profiles.base.Profile` instance or a string referencing a profile in the
            model's registry (:py:attr:`profiles`).

            .. hint::

                To be a valid :py:class:`~pisces.profiles.base.Profile` instance, the profile must have the same
                axes as :py:attr:`coordinate_system` or have axes which are a subset of them.

        field_name : str, optional
            The name to give to the newly generated field. If ``profile`` is a string and this is not filled, then
            the new field will have the same name as ``profile``. Otherwise, ``field_name`` is required.

            .. note::

                The ``field_name`` will be the location in the HDF5 file as well (``FIELDS/field_name``).

        chunking : bool, optional
            If `True`, evaluate the function in chunks. Default is `False`.

            .. tip::

                This is generally not necessary unless you cannot load the entire base grid into memory at once. This
                is particularly common if the function is operating in 3 or more dimensions, in which case even moderately
                resolved grids may take up significant memory.

        units : Optional[str], optional
            The units to give to the field. If ``units`` is not provided, then it is assumed that the field is
            dimensionless.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.

        Raises
        ------
        ValueError
            If the function, axes, or other parameters are invalid.
        """
        __logging__ = kwargs.pop("logging", True)
        # Setup the profile. If we got an actual profile, we need to check for the name
        # in the kwargs. Otherwise we need to look up the profile using the string we have.
        if isinstance(profile, Profile):
            if field_name is None:
                raise ValueError(
                    "`profile` argument was a Profile class, `profile_name` is a required kwarg."
                )
        elif isinstance(profile, str):
            try:
                if field_name is None:
                    field_name = profile

                profile = self.profiles[profile]
            except KeyError as e:
                raise KeyError(f"Profile `{field_name}` not found.") from e
        else:
            raise TypeError("`profile` must be either a string or a Profile class.")

        # Now pass along to the grid manager to actually add the field.
        self.grid_manager.add_field_from_profile(profile, field_name, **kwargs)

        # Produce the output log.
        if __logging__:
            self.logger.debug(
                "[SLVR] Field '%s' added from internal profile.", field_name
            )

    def convert_profile_to_field(
        self,
        profile_name: str,
        field_name: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Convert a profile to a model field.

        Ensures that a specified profile in :py:attr:`profiles` is converted into a field within the model.
        If ``field_name`` is not provided, it defaults to ``profile_name``.

        Parameters
        ----------
        profile_name : str
            The name of the profile to convert to a field. Must exist within :py:attr:`profiles`.

        field_name : str, optional
            The desired name for the resulting field. Defaults to ``profile_name`` if not provided.

        overwrite : bool, default=False
            If ``True``, existing field with the same name will be overwritten. If ``False`` and the
            field already exists, a ``ValueError`` is raised.

        Raises
        ------
        ValueError
            If ``profile_name`` does not exist in :py:attr:`profiles` or if ``overwrite`` is ``False`` and
            the field already exists.

        """
        # Look up the provided profile and ensure that it exists as expected.
        if profile_name not in self.profiles:
            raise ValueError(
                f"The profile `{profile_name}` does not appear to be present in {self}."
            )
        profile = self.profiles[profile_name]

        # Validate the field name. If the field name is not provided, we assume it
        # matches the profile name.
        if field_name is None:
            field_name = profile_name

        # Look up the correct units for the field based on that profile.
        _units = self.get_default_units(field_name)

        # Add the profile directly
        self.add_field_from_profile(
            profile,
            field_name=field_name,
            chunking=False,
            units=str(_units),
            logging=False,
            overwrite=overwrite,
        )
        self.logger.debug(
            "[EXEC] \t\tAdded field `%s` (units=%s) from profile.",
            field_name,
            str(_units),
        )


class _RadialModel(Model):
    r"""
    Private extended base class of the standard model class for models
    which have a default radial symmetry. Provides basic utility functions for
    descendant methods which inherit from this model.
    """
    # @@ VALIDATION MARKERS @@ #
    # These validation markers are used by the Model to constrain the valid
    # parameters for the model. Subclasses can modify the validation markers
    # to constrain coordinate system compatibility.
    #
    # : _IS_ABC : marks whether the model should seek out pathways or not.
    # : _INHERITS_PATHWAYS: will allow subclasses to inherit the pathways of their parent class.
    _IS_ABC: bool = True
    _INHERITS_PATHWAYS: bool = False

    # @@ CLASS PARAMETERS @@ #
    # The class parameters define several "standard" behaviors for the class.
    # These can be altered in subclasses to produce specific behaviors.
    DEFAULT_COORDINATE_SYSTEM = SphericalCoordinateSystem
    INIT_FREE_AXES = ["r"]

    # @@ UTILITY FUNCTIONS @@ #
    # These utility functions are used throughout the model generation process for various things
    # and are not sufficiently general to be worth implementing elsewhere.
    def get_radii(self) -> unyt.unyt_array:
        """
        Retrieve the radial coordinates in the appropriate length units.
        """
        return unyt.unyt_array(
            self.grid_manager.get_coordinates(axes=["r"]).ravel(),
            self.grid_manager.length_unit,
        )

    def construct_radial_spline(
        self,
        field_name: str,
        radii: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
    ):
        """
        Generate a radial spline for the specified field.

        Parameters
        ----------
        field_name : str
            Name of the field to spline.
        radii : unyt.unyt_array, optional
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
        # Validate the input field name and fetch the necessary field data and
        # base units. Ensure that field exists and that it is actually a radial field.
        if field_name not in self.FIELDS:
            raise ValueError(f"Field '{field_name}' does not exist in the model.")
        if set(self.FIELDS[field_name].AXES) != {"r"}:
            raise ValueError(
                f"Field '{field_name}' is not defined over the radial axis."
            )

        # Manage the radii construction as needed based on the input.
        if radii is None:
            radii = self.get_radii().d
        else:
            radii = radii.d if hasattr(radii, "units") else radii

        field_data = self.FIELDS[field_name][...].ravel()
        return InterpolatedUnivariateSpline(radii, field_data.d)

    def integrate_radial_density_field(
        self,
        density_field: str,
        mass_field: Optional[str] = None,
        mass_unit: Optional[Union[str, unyt.Unit]] = None,
        create_field: bool = False,
        overwrite: bool = False,
    ):
        """
        Integrate a radial density field to compute the enclosed mass profile.

        This method takes a specified radial density field (e.g., gas, stellar, dark matter density),
        integrates it over the radial grid, and computes the enclosed mass profile. The resulting mass
        profile is returned as a `unyt.unyt_array` with appropriate units. Optionally, the computed mass
        can be added to the model's field container.

        Parameters
        ----------
        density_field : str
            Name of the density field to integrate. Must be one of:
            ['total_density', 'gas_density', 'stellar_density', 'dark_matter_density'].

        mass_field : str, optional
            Name of the mass field to create. If not provided, it is retrieved from the model's
            configuration under `fields.{density_field}.mass_field`.

        mass_unit : str or unyt.Unit, optional
            Units for the mass field. If not provided, it is retrieved from the model's configuration
            under `fields.{density_field}.mass_unit`.

        create_field : bool, default=False
            If `True`, the integrated mass field will be added to the model's field container (`self.FIELDS`).
            Defaults to `False`.

        overwrite : bool, default=False
            If `True`, overwrite an existing mass field with the same name. If `False` and the field
            already exists, a `ValueError` is raised. Defaults to `False`.

        Returns
        -------
        unyt.unyt_array
            The integrated mass profile with appropriate units.

        Raises
        ------
        ValueError
            If `density_field` does not exist in the model or is not defined over the radial ('r') axis.

        KeyError
            If `mass_field` cannot be determined from the configuration and is not provided, or if
            `mass_unit` is missing.

        Notes
        -----
        This method assumes that the specified `density_field` is defined over the radial axis ('r'). The
        integration is performed using the coordinate system's `integrate_in_shells` method, which should
        handle the specifics of the geometry.

        """
        # Retrieve and validate the input field. It must be a valid radial field. Additionally, if we cannot locate
        # a mass field, we need to be given the mass field name and the units.
        if density_field not in self.FIELDS:
            raise ValueError(
                f"Density field `{density_field}` does not exist in {self}."
            )
        if set(self.FIELDS[density_field].AXES) != {"r"}:
            raise ValueError(
                f"Field '{density_field}' is not defined over the radial ('r') axis."
            )

        # Grab the density field.
        density_field_name = density_field
        density_field = self.FIELDS[density_field]

        # Look for a mass field / mass units. If we cannot find them, we need to raise errors.
        if mass_field is None:
            # We didn't get a mass field so we need to look it up and try to grab the default units.
            try:
                mass_field = self.config[f"fields.{density_field_name}.mass_field"]
                mass_unit = self.get_default_units(mass_field)
            except KeyError:
                raise KeyError(
                    f"Mass field for '{density_field_name}' could not be found in configuration or was "
                    f"missing units / reference to a mass field.\nIf it is not configured, it should be provided manually."
                )
        elif mass_unit is None:
            # We still need to look up the mass unit.
            try:
                mass_unit = self.get_default_units(mass_field)
            except KeyError:
                raise KeyError(
                    f"Mass field for '{density_field_name}' could not be found in configuration or was "
                    f"missing units.\nIf it is not configured, it should be provided manually."
                )
        else:
            # We have everything provided by the input arguments.
            pass

        # Construct the coordinates, build the interpolated spline and then
        # perform the shell integration routine. This should be adaptable
        # for any radial coordinate system due to the implementation of
        # integrate_in_shells.
        radii, spline = self.get_radii(), self.construct_radial_spline(
            density_field_name
        )

        # noinspection PyUnresolvedReferences
        # We can skip inspection here because we know coordinate system is a subclass of CoordinateSystem.
        enclosed_mass = self.coordinate_system.integrate_in_shells(spline, radii.d)

        # Manage the boundary estimate. We assume constant density and perform a second integrate in shells to
        # obtain the total enclosed.
        interior_density = spline(radii.d[0])
        integrand = lambda _r: interior_density * np.ones_like(_r)
        # noinspection PyUnresolvedReferences
        enclosed_mass += self.coordinate_system.integrate_in_shells(
            integrand, [0, radii.d[0]]
        )[1]

        # Manage the units. We compute `mass_units` which is the natural unit of the
        # computation and then covert to a target unit.
        base_mass_unit = (
            density_field.units * unyt.Unit(self.grid_manager.length_unit) ** 3
        )
        enclosed_mass = unyt.unyt_array(enclosed_mass, base_mass_unit).to(mass_unit)

        # Optionally add the computed mass field to the field container
        if create_field:
            self.FIELDS.add_field(
                mass_field,
                data=enclosed_mass,
                units=str(mass_unit),
                overwrite=overwrite,
                axes=["r"],
            )

        return enclosed_mass

    def compute_spherical_density_from_mass(
        self,
        mass_field: str,
        density_field: Optional[str] = None,
        density_unit: Optional[Union[str, unyt.Unit]] = None,
        create_field: bool = False,
        overwrite: bool = False,
    ):
        """
        Differentiate a radial mass field to compute the density profile.

        This method takes a specified radial mass field (e.g., gas, stellar, dark matter mass),
        differentiates it over the radial grid, and computes the enclosed density profile.

        Parameters
        ----------
        mass_field : str
            Name of the mass field to integrate. Must be one of:
            ['total_mass', 'gas_mass', 'stellar_mass', 'dark_matter_mass'].

        density_field : str, optional
            Name of the density field to create. If not provided, it is retrieved from the model's
            configuration under by searching for a corresponding field linked to the ``mass_field``.

        density_unit : str or unyt.Unit, optional
            Units for the density field. If not provided, it is retrieved from the model's configuration
            under `fields.{density_field}.unit`.

        create_field : bool, default=False
            If `True`, the density field will be added to the model's field container (`self.FIELDS`).
            Defaults to `False`.

        overwrite : bool, default=False
            If `True`, overwrite an existing density field with the same name. If `False` and the field
            already exists, a `ValueError` is raised. Defaults to `False`.

        Returns
        -------
        unyt.unyt_array
            The density profile with appropriate units.
        """
        # Retrieve and validate the input field. It must be a valid radial field. Additionally, if we cannot locate
        # a mass field, we need to be given the mass field name and the units.
        if mass_field not in self.FIELDS:
            raise ValueError(f"Mass field `{mass_field}` does not exist in {self}.")
        if set(self.FIELDS[mass_field].AXES) != {"r"}:
            raise ValueError(
                f"Field '{mass_field}' is not defined over the radial ('r') axis."
            )
        if self.coordinate_system.__class__.__name__ != "SphericalCoordinateSystem":
            raise ValueError(
                "The `compute_spherical_density_from_mass` method only works for SphericalCoordinateSystems."
            )

        # Grab the mass field.
        mass_field_name = mass_field
        mass_field = self.FIELDS[mass_field]

        # Look for a density field / density units. If we cannot find them, we need to raise errors.
        if density_field is None:
            # We didn't get a density field so we need to look it up and try to grab the default units.
            try:
                _valid_dfields = [
                    field
                    for field in self.FIELDS
                    if self.config["fields.{field}.mass_field"] == mass_field_name
                ]

                if len(_valid_dfields) != 1:
                    raise KeyError()

                density_field = _valid_dfields[0]
                density_unit = self.get_default_units(density_field)
            except KeyError:
                raise KeyError(
                    f"Mass field for '{density_field}' could not be found in configuration or was "
                    f"missing units / reference to a density field. \nIf it is not configured, it should be provided manually."
                )
        elif density_unit is None:
            # We still need to look up the density unit.
            try:
                density_unit = self.get_default_units(density_field)
            except KeyError:
                raise KeyError(
                    f"Mass field for '{density_field}' could not be found in configuration or was "
                    f"missing units.\nIf it is not configured, it should be provided manually."
                )
        else:
            # We have everything provided by the input arguments.
            pass

        # Construct the coordinates, build the interpolated spline and then
        # perform the shell integration routine. This should be adaptable
        # for any radial coordinate system due to the implementation of
        # integrate_in_shells.
        radii, spline = self.get_radii(), self.construct_radial_spline(density_field)

        # Compute the density by taking the derivative
        density_data = spline(radii.d, 1) / (4 * np.pi * radii.d**2)

        # Manage the units. We compute `mass_units` which is the natural unit of the
        # computation and then covert to a target unit.
        base_density_unit = (
            mass_field.units / unyt.Unit(self.grid_manager.length_unit) ** 3
        )
        density_data = unyt.unyt_array(density_data, base_density_unit).to(density_unit)

        # Optionally add the computed mass field to the field container
        if create_field:
            self.FIELDS.add_field(
                density_field,
                data=density_data,
                units=str(density_unit),
                overwrite=overwrite,
                axes=["r"],
            )

        return density_data
