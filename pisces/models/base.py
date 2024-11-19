from collections import OrderedDict
from pathlib import Path
from typing import Union, Type, List, Dict, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from grids.grid_base import Grid
from pisces.geometry.handlers import GeometryHandler
from pisces.grids.manager_base import GridManager
from pisces.profiles import Profile
from pisces.profiles.collections import HDF5ProfileRegistry

if TYPE_CHECKING:
    pass

# noinspection PyAttributeOutsideInit
class Model:
    """
    Base class for managing physical models with associated grids, profiles, and geometry.

    Class Attributes
    ----------------
    ALLOWED_COORDINATE_SYSTEMS : Optional[List[str]]
        A list of allowed coordinate system class names. If None, any coordinate system is permitted.
    ALLOWED_SYMMETRIES : Optional[List[str]]
        A list of allowed symmetry class names. If None, any symmetry is permitted.
    ALLOWED_GRID_MANAGER_CLASSES : Optional[List[str]]
        A list of allowed grid manager class names. If None, any grid manager class is permitted.
    """
    # Validation markers.
    # These can be adjusted to prevent users from attempting to
    # instantiate a model with invalid geometries / coordinate systems.
    ALLOWED_COORDINATE_SYSTEMS: Optional[List[str]] = None
    ALLOWED_SYMMETRIES: Optional[List[str]] = None
    ALLOWED_GRID_MANAGER_CLASSES: Optional[List[str]] = None

    # Solver classes
    # These are simply the solvers attached to this model. We load
    # the solver using the key value (saved in attrs of HDF5 file) and
    # instantiate on this model instance.
    SOLVERS = {}


    def __init__(self, path: Union[str, Path]):
        # Validate the path and ensure it exists
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Failed to find path: {self.path}")

        # Load essential components
        self._load_components()

    def _load_components(self):
        """
        Load the essential components of the model.
        """
        try:
            self._load_grid_manager()
            self._load_scratch_space()
            self._load_profiles()
            self._load_geometry()
            self._load_solver()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Model from path: {self.path}") from e

    def _load_grid_manager(self):
        """
        Load the grid manager from disk.
        """
        try:
            self._grid_manager = GridManager.load_subclass_from_disk(self.path)
            self._check_allowed_grid_manager(self._grid_manager.__class__)
        except Exception as e:
            raise ValueError(f"Failed to load GridManager from path: {self.path}") from e

    @classmethod
    def _check_allowed_grid_manager(cls,manager_class: Type[GridManager]):
        """
        Check if the loaded GridManager class is allowed.
        """
        if cls.ALLOWED_GRID_MANAGER_CLASSES is not None:
            if manager_class.__name__ not in cls.ALLOWED_GRID_MANAGER_CLASSES:
                raise ValueError(f"GridManager class '{manager_class.__name__}' is not allowed for model type '{cls.__name__}'. "
                                 f"Allowed classes: {cls.ALLOWED_GRID_MANAGER_CLASSES}")

    def _load_scratch_space(self):
        """
        Load or create the scratch space in the grid manager.
        """
        try:
            self._scratch = self.grid_manager.handle.require_group("SCRATCH")
        except Exception as e:
            raise RuntimeError("Failed to load or create scratch space.") from e

    def _load_profiles(self):
        """
        Load the profile registry from the grid manager.
        """
        try:
            profiles_handle = self.grid_manager.handle.require_group("PROFILES")
            self._profiles = HDF5ProfileRegistry(profiles_handle)
        except Exception as e:
            raise RuntimeError("Failed to load profile registry.") from e

    def _load_geometry(self):
        """
        Load the geometry handler from the grid manager.
        """
        try:
            geometry_handle = self.grid_manager.handle.get("GEOMETRY")
            if geometry_handle is None:
                raise KeyError("GEOMETRY group not found in the HDF5 structure.")
            self._geometry_handler = GeometryHandler.from_hdf5(geometry_handle)
            self._check_allowed_geometry(self.geometry_handler)
        except KeyError as e:
            raise ValueError("The grid manager lacks a 'GEOMETRY' group.") from e
        except Exception as e:
            raise RuntimeError("Failed to load the geometry handler.") from e

    def _load_solver(self):
        solver_name = self.grid_manager.handle.attrs.get('SOLVER',None)
        if solver_name is not None:
            try:
                _solver: Type[Solver] = self.__class__.SOLVERS[solver_name]
            except KeyError as e:
                raise ValueError(f"Solver {solver_name} is not a recognized solver class for model type '{self.__class__.__name__}'. "
                                 f"Was this model generated from a different model class?") from e

            # instantiate the solver
            try:
                _solver: Optional[Solver] = _solver(self)
            except Exception as e:
                raise ValueError(f"Failed to instantiate solver {solver_name} because of the following error: {e}.") from e
        else:
            _solver: Optional[Solver] = None

        self._solver = _solver

    @classmethod
    def _check_allowed_geometry(cls,geometry_handler: GeometryHandler):
        """
        Check if the loaded geometry handler uses allowed coordinate systems and symmetries.
        """
        if cls.ALLOWED_COORDINATE_SYSTEMS is not None:
            coord_system_name = geometry_handler.coordinate_system.__class__.__name__
            if coord_system_name not in cls.ALLOWED_COORDINATE_SYSTEMS:
                raise ValueError(f"Coordinate system '{coord_system_name}' is not allowed for model class '{cls.__name__}'. "
                                 f"Allowed systems: {cls.ALLOWED_COORDINATE_SYSTEMS}")

        if cls.ALLOWED_SYMMETRIES is not None:
            symmetry = {geometry_handler.coordinate_system.AXES[k] for k in geometry_handler.symmetry.symmetry_axes}
            if symmetry not in cls.ALLOWED_SYMMETRIES:
                raise ValueError(f"Symmetry '{symmetry}' is not allowed for model class '{cls.__name__}'. "
                                 f"Allowed symmetries: {cls.ALLOWED_SYMMETRIES}")

    def __str__(self):
        return f"<{self.__class__.__name__}: path={self.path}>"

    def __del__(self):
        """
        Ensure resources are released when the Model instance is deleted.
        """
        try:
            self.grid_manager.close()
        except AttributeError:
            pass  # Ignore if grid_manager was not fully initialized

    @property
    def grid_manager(self):
        return self._grid_manager

    @property
    def geometry_handler(self):
        return self._geometry_handler

    @property
    def profiles(self):
        return self._profiles

    @property
    def scratch_space(self):
        return self._scratch

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self,solver_name: str):
        # Resolve the solver class so that we can instantiate it.
        try:
            _solver_class = self.__class__.SOLVERS[solver_name]
        except KeyError as e:
            raise ValueError(f"Solver named '{solver_name}' is not a recognized solver type for model class '{self.__class__.__name__}'.")

        # Set the solver class
        try:
            self._solver = _solver_class(self)
        except Exception as e:
            raise ValueError(f"Failed to instantiate new solver (named {solver_name}) for {self}: {e}.") from e

        # Set the new solver
        self.grid_manager.handle.attrs['SOLVER'] = solver_name

    @classmethod
    def build_skeleton(cls,
                       path: Union[str, Path],
                       bbox: NDArray[np.floating],
                       grid_size: NDArray[np.int_],
                       grid_manager_class: Type[GridManager],
                       geometry_handler: GeometryHandler,
                       profiles: Optional[Dict[str, Profile]] = None,
                       solver: str = None,
                       axes: Optional[List[str]] = None,
                       overwrite: bool = False) -> "Model":
        """
        Build the skeleton structure for a new Model instance.

        This method initializes the necessary components of the model, including the grid manager,
        geometry handler, profiles, and scratch space, and saves them into an HDF5 file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the HDF5 file where the model will be saved.
        bbox : NDArray[np.floating]
            Bounding box for the grid structure. Must be a 2D array with shape `(2, NDIM)`.
        grid_size : NDArray[np.int_]
            Size of the grid along each dimension.
        grid_manager_class : Type[GridManager]
            The class used for managing grids, must be a subclass of `GridManager`.
        geometry_handler : GeometryHandler
            The geometry handler to initialize and save.
        profiles : dict[str, Profile], optional
            A dictionary of profile names to Profile objects to initialize in the profile registry.
        axes : list[str], optional
            Names of the axes for the grid. Defaults to `['X', 'Y', 'Z']` up to the grid dimensionality.
        overwrite : bool, default False
            If True, overwrites any existing file at the specified path.

        Returns
        -------
        Model
            An initialized `Model` instance pointing to the created HDF5 structure.

        Raises
        ------
        ValueError
            If the file already exists and `overwrite` is False.

        Examples
        --------

        Generating a generic 1D skeleton for a model:

        >>> from pisces.geometry.handlers import GeometryHandler
        >>> from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
        >>> handler = GeometryHandler(CartesianCoordinateSystem())
        >>> model = Model.build_skeleton('test.hdf5',[0,1], [100], GridManager, handler, overwrite=True)

        """
        path = Path(path)

        # Handle file existence
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise ValueError(f"File at path '{path}' already exists. Use `overwrite=True` to replace it.")

        # Default axes to match the dimensionality of the grid
        if axes is None:
            axes = ['X', 'Y', 'Z'][:len(grid_size)]

        # Step 1: Initialize the grid manager
        cls._check_allowed_grid_manager(grid_manager_class)
        grid_manager = grid_manager_class(
            path,
            axes=axes,
            bbox=bbox,
            grid_size=grid_size,
            overwrite=True
        )

        return cls.build_skeleton_on_grid_manager(grid_manager,
                                                  geometry_handler,
                                                  profiles = profiles,
                                                  solver = solver,
                                                  overwrite = overwrite)

    @classmethod
    def build_skeleton_on_grid_manager(cls,
                                       grid_manager: GridManager,
                                       geometry_handler: GeometryHandler,
                                       profiles: Optional[Dict[str, Profile]] = None,
                                       solver: str = None,
                                       overwrite: bool = False):
        cls._check_allowed_geometry(geometry_handler)
        try:
            # Step 2: Initialize the geometry handler
            geometry_group = grid_manager.handle.require_group("GEOMETRY")
            geometry_handler.to_hdf5(geometry_group)

            # Step 3: Initialize the profile registry
            profiles_group = grid_manager.handle.require_group("PROFILES")
            profile_registry = HDF5ProfileRegistry(profiles_group)

            if profiles:
                for profile_name, profile in profiles.items():
                    profile_registry.add_profile(profile_name, profile)

            # Step 4: Initialize the scratch space
            grid_manager.handle.require_group("SCRATCH")

            # Step 5: Add the solver
            if solver is not None:
                grid_manager.handle.attrs['SOLVER'] = str(solver)

        except Exception as e:
            # Ensure resources are closed if any step fails
            grid_manager.handle.flush()
            grid_manager.close()
            raise RuntimeError(f"Failed to build skeleton structure: {e}") from e

        # Step 5: Close the grid manager to flush data and finalize the structure
        grid_manager.handle.flush()
        path = grid_manager.path
        grid_manager.close()

        # Return the initialized model
        return cls(path)

class SolverMeta(type):
    def __new__(mcls, name, bases, attrs):
        # Create the superclass object as normal.
        cls = super().__new__(mcls, name, bases, attrs)

        # Generate pipeline registry
        cls.pipeline_registry = {
            attr_name: {
                "kwargs": getattr(attr_value, "_kwargs", {}),
            }
            for attr_name, attr_value in attrs.items()
            if getattr(attr_value, "_is_pipeline", False)
        }

        # Generate state checkers registry
        cls.state_checkers = {
            attr_value.name: {"name": attr_name, "type": attr_value.type}
            for attr_name, attr_value in attrs.items()
            if getattr(attr_value, "_is_state_checker", False)
        }

        return cls


class Solver(metaclass=SolverMeta):
    """
    Base class for solvers that operate on models and grids.

    Solvers consist of "pipelines" that process grids based on the model's state
    and the grid's attributes. Pipelines are automatically registered and validated
    against state checkers to ensure compatibility.

    Attributes
    ----------
    ALLOWED_MODEL_CLASSES : Optional[list[str]]
        List of allowed model class names for this solver. If None, all models are allowed.

        For developers, this should be changed in subclasses of :py:class:`Solver` to ensure
        that users cannot try to instantiate an invalid solver for a given model type.
    _model_valid_pipelines : OrderedDict
        Ordered dictionary of valid pipelines for the current model instance.

        This is a dictionary of ``key`` and ``kwarg`` matches for pipelines which have passed
        the model's checks but have not necessarily passed any grid checks.

    Parameters
    ----------
    model : Model
        The model instance to which the solver is bound.

    Notes
    -----
    - Pipelines must be decorated with `@pipeline`.
    - State checkers must be decorated with `@state_checker`.
    - The class-level attribute `ALLOWED_MODEL_CLASSES` can be used to restrict
      solver compatibility to specific model types.

    See Also
    --------
    pipeline : Decorator for defining pipeline functions.
    state_checker : Decorator for defining state checker functions.
    """
    ALLOWED_MODEL_CLASSES = None

    def __init__(self, model: 'Model'):
        self.model = model

        # Validate model compatibility
        if (self.__class__.ALLOWED_MODEL_CLASSES is not None and
                self.model.__class__.__name__ not in self.__class__.ALLOWED_MODEL_CLASSES):
            raise ValueError(f"Cannot initialize solver '{self.__class__.__name__}' for model type "
                             f"'{self.model.__class__.__name__}'. This solver does not support the model class.")

        # Determine valid pipelines
        self._model_valid_pipelines = self._construct_valid_pipelines()

    def _construct_valid_pipelines(self) -> OrderedDict:
        """
        Construct an ordered dictionary of valid pipelines based on state checkers.

        Returns
        -------
        OrderedDict
            An ordered dictionary where keys are valid pipeline names, and values
            are the corresponding kwargs that validate the pipeline.
        """
        valid_pipelines = OrderedDict()

        for pipeline_name, pipeline_data in self.__class__.pipeline_registry.items():
            if self.check_pipeline(pipeline_name, grid=None):
                valid_pipelines[pipeline_name] = pipeline_data['kwargs']

        return valid_pipelines

    def get_state(self, grid: Optional['Grid']=None) -> Dict[str, Optional[bool]]:
        """
        Retrieve the current state of the model and optionally the grid.

        Parameters
        ----------
        grid : Optional[Grid]
            The grid to check against state checkers. If None, only model-level
            state checkers are evaluated.

        Returns
        -------
        Dict[str, Optional[bool]]
            A dictionary of state checker results, with checker names as keys and
            boolean results (or None for uncheckable states) as values.
        """
        state = {}
        for name, checker in self.__class__.state_checkers.items():
            if checker['type'] == 'grid':
                state[name] = getattr(self, checker['name'])(grid) if grid else None
            else:
                state[name] = getattr(self, checker['name'])()

        return state

    def check_state(self, grid: Optional['Grid']=None, **kwargs) -> bool:
        """
        Validate the current state against expected state values.

        Parameters
        ----------
        grid : Optional[Grid]
            The grid to check against state checkers.
        kwargs : dict
            Expected state values to validate.

        Returns
        -------
        bool
            True if all state values match the expected values, False otherwise.
        """
        state = self.get_state(grid=grid)
        return all(state.get(key) == value for key, value in kwargs.items())

    def check_pipeline(self, pipeline: str, grid: Optional['Grid']=None) -> bool:
        """
        Validate whether a pipeline is compatible with the current state.

        Parameters
        ----------
        pipeline : str
            Name of the pipeline to validate.
        grid : Optional[Grid]
            The grid to check against state checkers.

        Returns
        -------
        bool
            True if the pipeline is valid, False otherwise.
        """
        pipeline_kwargs = self.pipeline_registry[pipeline]['kwargs']
        return self.check_state(grid=grid, **pipeline_kwargs)

    def has_pipeline(self,grid: 'Grid' = None):
        try:
            pipeline = self.get_pipeline(grid=grid)
        except Exception as e:
            raise ValueError(f"Failed to check pipeline: {e}.")

        return pipeline is not None

    def get_pipeline(self, grid: Optional['Grid']=None, __preferred__: Optional[str] = None) -> str:
        """
        Retrieve the name of a valid pipeline for the current state.

        Parameters
        ----------
        grid : Optional[Grid]
            The grid to check against state checkers.
        __preferred__ : Optional[str]
            Preferred pipeline name to prioritize.

        Returns
        -------
        str
            The name of a valid pipeline, or None if no valid pipelines exist.
        """
        valid_pipelines = [name for name in self.pipeline_registry if self.check_pipeline(name, grid=grid)]

        if not valid_pipelines:
            raise ValueError(
                f"No valid pipelines found in solver {self.__class__.__name__} for model {self.model} and grid {grid}. "
                f"Ensure that the model and grid satisfy the requirements for at least one pipeline.")

        return __preferred__ if __preferred__ in valid_pipelines else valid_pipelines[0]

    def list_pipelines(self) -> List[str]:
        """
        List the names of all registered pipelines.

        Returns
        -------
        List[str]
            A list of registered pipeline names.
        """
        return list(self.__class__.pipeline_registry.keys())

    def pipeline_status(self, grid: Optional['Grid'] = None) -> Dict[str,Dict[str,...]]:
        """
        Generate a table of pipelines, their conditions, and validation status.

        Parameters
        ----------
        grid : Optional[Grid]
            The grid to check against state checkers.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the pipeline names, their conditions, and whether they are valid.
        """
        state = self.get_state(grid=grid)

        rows = {}
        for pipeline_name, pipeline_data in self.__class__.pipeline_registry.items():
            conditions = pipeline_data['kwargs']
            valid = all(state.get(key) == value for key, value in conditions.items())
            rows[pipeline_name] = {
                "Conditions": conditions,
                "Valid": valid
            }

        return rows

    def solve(self, grid: 'Grid', __preferred__: Optional[str] = None):
        """
        Execute a valid pipeline on the given grid.

        Parameters
        ----------
        grid : Grid
            The grid to solve.
        __preferred__ : Optional[str]
            Preferred pipeline name to prioritize.

        Returns
        -------
        Any
            The result of the pipeline execution.

        Raises
        ------
        ValueError
            If no valid pipeline is found.
        """
        pipeline = self.get_pipeline(grid, __preferred__=__preferred__)

        if pipeline is None:
            raise ValueError("Failed to find a valid pipeline.")

        return getattr(self, pipeline)(grid)

    def __str__(self):
        return f"<{self.__class__.__name__}(model={self.model})>"

    def __repr__(self):
        return self.__str__()
