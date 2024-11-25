from collections import OrderedDict
from pathlib import Path
from typing import Union, Type, List, Dict, Optional, TYPE_CHECKING, Tuple, Any, Callable
import numpy as np
from numpy.typing import NDArray

from pisces.geometry import CoordinateSystem
from pisces.profiles import Profile
from pisces.profiles.collections import HDF5ProfileRegistry
from pisces.models.fields.base import ModelFieldContainer
from pisces.io.hdf5 import HDF5_File_Handle
import unyt
if TYPE_CHECKING:
    pass

# noinspection PyAttributeOutsideInit
class Model:
    """
    Base class for managing physical models with associated _grids, profiles, and geometry.

    Class Attributes
    ----------------
    ALLOWED_COORDINATE_SYSTEMS : Optional[List[str]]
        A list of allowed coordinate system class names. If None, any coordinate system is permitted.
    ALLOWED_SYMMETRIES : Optional[List[str]]
        A list of allowed symmetry class names. If None, any symmetry is permitted.
    """
    # Validation markers.
    # These can be adjusted to prevent users from attempting to
    # instantiate a model with invalid geometries / coordinate systems.
    # Models may support multiple coordinate systems or symmetries if they can
    # be solved generically by the solvers available.
    ALLOWED_COORDINATE_SYSTEMS: Optional[List[str]] = None

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

        # Create the handle reference.
        self._handle = HDF5_File_Handle(self.path,'r+')

        # Load essential components
        self._load_components()

        # Load the FIELDS
        _field_handle = self._handle.require_group("FIELDS")
        self.FIELDS = ModelFieldContainer(self)

    def _load_components(self):
        """
        Load the essential components of the model.
        """
        try:
            self._load_scratch_space()
            self._load_profiles()
            self._load_coordinate_system()
            self._load_solver()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Model from path: {self.path}") from e

    def _load_scratch_space(self):
        """
        Load or create the scratch space in the grid manager.
        """
        try:
            self._scratch = self._handle.require_group("SCRATCH")
        except Exception as e:
            raise RuntimeError("Failed to load or create scratch space.") from e

    def _load_profiles(self):
        """
        Load the profile registry from the grid manager.
        """
        try:
            profiles_handle = self._handle.require_group("PROFILES")
            self._profiles = HDF5ProfileRegistry(profiles_handle)
        except Exception as e:
            raise RuntimeError("Failed to load profile registry.") from e

    def _load_coordinate_system(self):
        """
        Load the geometry handler from the grid manager.
        """
        try:
            coord_handle = self._handle.get("CSYS")
            if coord_handle is None:
                raise KeyError("CSYS group not found in the HDF5 structure.")
            self._coordinate_system = CoordinateSystem.from_file(coord_handle,fmt='hdf5')
            self._check_allowed_geometry(self._coordinate_system)
        except KeyError as e:
            raise ValueError("The grid manager lacks a 'CSYS' group.") from e
        except Exception as e:
            raise RuntimeError("Failed to load the geometry handler.") from e

    def _load_solver(self):
        solver_name = self._handle.attrs.get('SOLVER',None)
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
    def _check_allowed_geometry(cls,csys: CoordinateSystem):
        """
        Check if the loaded geometry handler uses allowed coordinate systems and symmetries.
        """
        if cls.ALLOWED_COORDINATE_SYSTEMS is not None:
            coord_system_name = csys.__class__.__name__
            if coord_system_name not in cls.ALLOWED_COORDINATE_SYSTEMS:
                raise ValueError(f"Coordinate system '{coord_system_name}' is not allowed for model class '{cls.__name__}'. "
                                 f"Allowed systems: {cls.ALLOWED_COORDINATE_SYSTEMS}")


    def __str__(self):
        return f"<{self.__class__.__name__}: path={self.path}>"

    def __del__(self):
        """
        Ensure resources are released when the Model instance is deleted.
        """
        try:
            self._handle.close()
        except AttributeError:
            pass  # Ignore if grid_manager was not fully initialized


    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._coordinate_system

    @property
    def handle(self) -> HDF5_File_Handle:
        return self._handle

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
        self._handle.attrs['SOLVER'] = solver_name

    # @classmethod
    # def build_skeleton(cls,
    #                    path: Union[str, Path],
    #                    bbox: NDArray[np.floating],
    #                    grid_size: NDArray[np.int_],
    #                    grid_manager_class: Type[GridManager],
    #                    geometry_handler: GeometryHandler,
    #                    profiles: Optional[Dict[str, Profile]] = None,
    #                    solver: str = None,
    #                    axes: Optional[List[str]] = None,
    #                    overwrite: bool = False) -> "Model":
    #     """
    #     Build the skeleton structure for a new Model instance.
    #
    #     This method initializes the necessary components of the model, including the grid manager,
    #     geometry handler, profiles, and scratch space, and saves them into an HDF5 file.
    #
    #     Parameters
    #     ----------
    #     path : Union[str, Path]
    #         Path to the HDF5 file where the model will be saved.
    #     bbox : NDArray[np.floating]
    #         Bounding box for the grid structure. Must be a 2D array with shape `(2, NDIM)`.
    #     grid_size : NDArray[np.int_]
    #         Size of the grid along each dimension.
    #     grid_manager_class : Type[GridManager]
    #         The class used for managing _grids, must be a subclass of `GridManager`.
    #     geometry_handler : GeometryHandler
    #         The geometry handler to initialize and save.
    #     profiles : dict[str, Profile], optional
    #         A dictionary of profile names to Profile objects to initialize in the profile registry.
    #     axes : list[str], optional
    #         Names of the axes for the grid. Defaults to `['X', 'Y', 'Z']` up to the grid dimensionality.
    #     overwrite : bool, default False
    #         If True, overwrites any existing file at the specified path.
    #
    #     Returns
    #     -------
    #     Model
    #         An initialized `Model` instance pointing to the created HDF5 structure.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the file already exists and `overwrite` is False.
    #
    #     Examples
    #     --------
    #
    #     Generating a generic 1D skeleton for a model:
    #
    #     >>> from pisces.geometry.handlers import GeometryHandler
    #     >>> from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
    #     >>> handler = GeometryHandler(CartesianCoordinateSystem())
    #     >>> model = Model.build_skeleton('test.hdf5',[0,1], [100], GridManager, handler, overwrite=True)
    #
    #     """
    #     path = Path(path)
    #
    #     # Handle file existence
    #     if path.exists():
    #         if overwrite:
    #             path.unlink()
    #         else:
    #             raise ValueError(f"File at path '{path}' already exists. Use `overwrite=True` to replace it.")
    #
    #     # Default axes to match the dimensionality of the grid
    #     if axes is None:
    #         axes = ['X', 'Y', 'Z'][:len(grid_size)]
    #
    #     # Step 1: Initialize the grid manager
    #     cls._check_allowed_grid_manager(grid_manager_class)
    #     grid_manager = grid_manager_class(
    #         path,
    #         axes=axes,
    #         bbox=bbox,
    #         grid_size=grid_size,
    #         overwrite=True
    #     )
    #
    #     return cls.build_skeleton_on_grid_manager(grid_manager,
    #                                               geometry_handler,
    #                                               profiles = profiles,
    #                                               solver = solver,
    #                                               overwrite = overwrite)
    #
    # @classmethod
    # def build_skeleton_on_grid_manager(cls,
    #                                    grid_manager: GridManager,
    #                                    geometry_handler: GeometryHandler,
    #                                    profiles: Optional[Dict[str, Profile]] = None,
    #                                    solver: str = None,
    #                                    overwrite: bool = False):
    #     cls._check_allowed_geometry(geometry_handler)
    #     try:
    #         # Step 2: Initialize the geometry handler
    #         geometry_group = grid_manager.handle.require_group("GEOMETRY")
    #         geometry_handler.to_hdf5(geometry_group)
    #
    #         # Step 3: Initialize the profile registry
    #         profiles_group = grid_manager.handle.require_group("PROFILES")
    #         profile_registry = HDF5ProfileRegistry(profiles_group)
    #
    #         if profiles:
    #             for profile_name, profile in profiles.items():
    #                 profile_registry.add_profile(profile_name, profile)
    #
    #         # Step 4: Initialize the scratch space
    #         grid_manager.handle.require_group("SCRATCH")
    #
    #         # Step 5: Add the solver
    #         if solver is not None:
    #             grid_manager.handle.attrs['SOLVER'] = str(solver)
    #
    #     except Exception as e:
    #         # Ensure resources are closed if any step fails
    #         grid_manager.handle.flush()
    #         grid_manager.close()
    #         raise RuntimeError(f"Failed to build skeleton structure: {e}") from e
    #
    #     # Step 5: Close the grid manager to flush data and finalize the structure
    #     grid_manager.handle.flush()
    #     path = grid_manager.path
    #     grid_manager.close()
    #
    #     # Return the initialized model
    #     return cls(path)

class SolverMeta(type):
    def __new__(mcls, name, bases, attrs):
        # Create the superclass object as normal.
        cls = super().__new__(mcls, name, bases, attrs)

        # Register all of the nodes
        cls.NODES = {
            attr_value.name: attr_name for attr_name,attr_value in attrs.items()
            if getattr(attr_value,'_is_node',False)
        }

        # Register all of the CONDITIONS
        # All of the conditions are collected as "name": name=attr_name, type=attr_value.typ.
        cls.CONDITIONS = {
            attr_value.name: dict(name=attr_name,type=attr_value.type) for attr_name,attr_value in attrs.items()
            if getattr(attr_value,'_is_condition',False)
        }

        return cls

class _SolverNode:
    """
    Represents a node in the solver's Directed Acyclic Graph (DAG).

    Each node is linked to a specific method (or function reference) within the solver
    and is responsible for executing a logical step in the computational pipeline.

    Attributes
    ----------
    name : str
        The name of the node, used for identification and referencing in the DAG.
    solver : Solver
        The solver instance to which this node belongs.
    ref : Optional[str]
        The name of the method in the solver to execute when this node is called.
        If None, the node acts as a pass-through and always returns True.

    Methods
    -------
    __call__(grid)
        Executes the node's logic by invoking the associated method in the solver.
        If no method is associated (ref is None), it returns True by default.
    """

    def __init__(self, name: str, solver: 'Solver', ref: Optional[str] = None):
        """
        Initialize a solver node.

        Parameters
        ----------
        name : str
            The unique name of the node.
        solver : Solver
            The solver instance to which this node belongs.
        ref : Optional[str], optional
            The name of the method in the solver to execute when this node is called.
            If None, the node acts as a pass-through node.

        Raises
        ------
        ValueError
            If the provided `ref` does not correspond to a valid method in the solver.
        """
        self.name = name
        self.solver = solver
        self.ref = ref

        # Validate that the referenced method exists in the solver
        if self.ref is not None and not hasattr(self.solver, self.ref):
            raise ValueError(f"Node '{self.name}' references an undefined solver method '{self.ref}'.")

    def __call__(self, grid):
        """
        Execute the node's logic by invoking the associated method in the solver.

        Parameters
        ----------
        grid : Grid
            The computational grid on which the node's logic operates.

        Returns
        -------
        Any
            The result of the associated solver method's execution.
            If no method is associated (ref is None), returns True.

        Raises
        ------
        RuntimeError
            If the associated solver method raises an exception during execution.
        """
        if self.ref is None:
            # If no method reference is provided, return True as a pass-through.
            return True

        try:
            # Look up the solver method and execute it with the grid.
            method = getattr(self.solver, self.ref)
            return method(grid)
        except Exception as e:
            raise RuntimeError(f"Error executing node '{self.name}' with method '{self.ref}': {e}") from e

    def __repr__(self):
        """
        Return a string representation of the _SolverNode.

        Returns
        -------
        str
            A string representation containing the node name and associated method reference.
        """
        return f"<_SolverNode(name={self.name}, ref={self.ref})>"

class _SolverEdge:
    """
    Represents an edge in the solver's Directed Acyclic Graph (DAG).

    An edge defines a dependency between two nodes (`start` and `end`) and optionally
    includes a condition for execution. The condition is evaluated using a method
    (or function reference) in the solver.

    Attributes
    ----------
    start : str
        The name of the starting node.
    end : str
        The name of the ending node.
    solver : Solver
        The solver instance to which this edge belongs.
    ref : Optional[str]
        The name of the method in the solver to evaluate the edge condition.
        If None, the edge condition always evaluates to True.

    Methods
    -------
    __call__(grid, result)
        Evaluates the edge condition using the associated method in the solver.
        If no method is associated (ref is None), it always returns True.
    """

    def __init__(self, start: str, end: str, solver: 'Solver', ref: Optional[str] = None,typ: str = 'model'):
        """
        Initialize a solver edge.

        Parameters
        ----------
        start : str
            The name of the starting node.
        end : str
            The name of the ending node.
        solver : Solver
            The solver instance to which this edge belongs.
        ref : Optional[str], optional
            The name of the method in the solver to evaluate the edge condition.
            If None, the edge condition always evaluates to True.
        typ: Literal['model','grid']
            The type of condition. If the condition is a model condition then it
            does not rely on the grid or the result of the previous node, just on
            the model at init. This allows the DAG to be reduced at __init__ by removing
            edges which fail their model conditions.
        Raises
        ------
        ValueError
            If the provided `ref` does not correspond to a valid method in the solver.
        """
        self.start = start
        self.end = end
        self.solver = solver
        self.ref = ref
        self.type = typ

        # Validate that the referenced method exists in the solver
        if self.ref is not None and not hasattr(self.solver, self.ref):
            raise ValueError(f"Edge from '{self.start}' to '{self.end}' references an undefined condition method '{self.ref}'.")

    def __call__(self, grid=None, result=None):
        """
        Evaluate the edge condition.

        Parameters
        ----------
        grid : Grid
            The computational grid on which the edge condition operates.
        result : Any
            The result produced by the starting node (`start`), used as input to the condition.

        Returns
        -------
        bool
            True if the edge condition is satisfied, False otherwise.

        Raises
        ------
        RuntimeError
            If the associated solver method raises an exception during execution.
        """
        if self.ref is None:
            # If no method reference is provided, the edge is always valid.
            return True

        try:
            # Look up the solver method and evaluate the condition.
            method = getattr(self.solver, self.ref)
            if self.type == 'model':
                return method()
            else:
                return method(grid, result)
        except Exception as e:
            raise RuntimeError(f"Error evaluating edge condition for '{self.start} -> {self.end}' with method '{self.ref}': {e}") from e

    def __repr__(self):
        """
        Return a string representation of the _SolverEdge.

        Returns
        -------
        str
            A string representation containing the edge details and associated method reference.
        """
        return f"<_SolverEdge(start={self.start}, end={self.end}, ref={self.ref})>"


class Solver(metaclass=SolverMeta):
    EDGES: Dict[Tuple[str,str],str] = {}
    ALLOWED_MODEL_CLASSES = None

    def __init__(self, model: 'Model'):
        self.model = model

        # Validate model compatibility
        self._validate_model()

        # Construct nodes and edges
        self._nodes: Dict[str, _SolverNode] = {}
        self._edges: Dict[Tuple[str, str], _SolverEdge] = {}
        self._construct_nodes()
        self._construct_edges()
        self._prune_disconnected_graph()

    def _validate_model(self):
        """
        Validate that the model is compatible with this solver.
        """
        if (self.__class__.ALLOWED_MODEL_CLASSES is not None and
                self.model.__class__.__name__ not in self.__class__.ALLOWED_MODEL_CLASSES):
            raise ValueError(f"Cannot initialize solver '{self.__class__.__name__}' for model type "
                             f"'{self.model.__class__.__name__}'. This solver does not support the model class.")

    def _construct_nodes(self):
        """
        Construct _SolverNode instances from the class-level NODES attribute.
        """
        # Add all nodes from the metaclass
        for node_name, ref in self.__class__.NODES.items():
            self._nodes[node_name] = _SolverNode(name=node_name, solver=self, ref=ref)

        # Add implicit 'start' and 'end' nodes
        self._nodes["start"] = _SolverNode(name="start", solver=self, ref=None)
        self._nodes["end"] = _SolverNode(name="end", solver=self, ref=None)

    def _construct_edges(self):
        """
        Construct _SolverEdge instances from the class-level EDGES attribute.

        This method uses the metaclass-registered CONDITIONS to evaluate edge conditions.
        """
        valid_edges = {}

        for (start,end), ref in self.__class__.EDGES.items():
            if start not in self._nodes or end not in self._nodes:
                raise ValueError(f"Invalid edge: nodes '{start}' or '{end}' do not exist.")

            # Create the edge
            edge = _SolverEdge(start,
                               end,
                               self,
                               ref=self.__class__.CONDITIONS[ref]['name'],
                               typ=self.__class__.CONDITIONS[ref]['type'])

            # Evaluate model conditions during graph construction
            if edge.type == "model" and not edge():  # Model conditions don't use grid or result
                continue

            valid_edges[(start, end)] = edge

        self._edges = valid_edges

    def _prune_disconnected_graph(self):
        """
        Remove nodes and edges that are not connected to the 'start' node.
        """
        reachable_nodes = self._find_all_reachable_nodes('start')

        self._nodes = {name: node for name, node in self._nodes.items() if name in reachable_nodes}
        self._edges = {key: edge for key, edge in self._edges.items() if key[0] in reachable_nodes and key[1] in reachable_nodes}

    def _find_all_reachable_nodes(self,start: str):
        reachable_nodes = set()

        for (_start,_end), _ in self._edges.items():
            if start == _start:
                reachable_nodes.add(_start)
                if _end not in reachable_nodes:
                    reachable_nodes = reachable_nodes.union(self._find_all_reachable_nodes(_end))

        return reachable_nodes


    def __repr__(self):
        """
        String representation of the solver's current DAG structure.
        """
        nodes = ', '.join(self._nodes.keys())
        edges = ', '.join(f"{start} -> {end}" for start, end in self._edges)
        return f"<Solver(nodes=[{nodes}], edges=[{edges}])>"

if __name__ == '__main__':
    from pisces.geometry import SphericalCoordinateSystem, GeometryHandler, Symmetry
    coord_system = SphericalCoordinateSystem()
    import h5py
    with h5py.File('test.hdf5','w') as fio:
        fio.require_group('CSYS')
        coord_system.to_file(fio['CSYS'],'hdf5')

    from pisces.profiles import NFWDensityProfile

    q = NFWDensityProfile(rho_0=1e5,r_s=200)
    tq = NFWDensityProfile(rho_0=1e6, r_s=200)

    rr = np.logspace()