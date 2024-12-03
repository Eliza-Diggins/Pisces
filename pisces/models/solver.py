import h5py
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pisces.models.base import Model


# noinspection PyUnresolvedReferences,PyProtectedMember
class ModelSolver:
    """
    Solver class for executing and managing pathways in a model.

    Parameters
    ----------
    model : Model
        The model instance this solver is associated with.
    is_solved : bool, optional
        Tracks whether the solver has been initialized or solved (default: False).
    default : str, optional
        Default pathway to execute if none is specified (default: None).

    Attributes
    ----------
    model : Model
        The associated model instance.
    is_solved : bool
        Tracks whether the solver has been initialized or solved.
    default : str
        Default pathway to execute.
    """

    def __init__(self, model: 'Model', is_solved: bool = False, default: Optional[str] = None):
        self.model: 'Model' = model
        self.is_solved: bool = is_solved
        self.default: Optional[str] = default

    def __call__(self, pathway: Optional[str] = None, overwrite: bool = False):
        """
        Execute a pathway on the model.

        Parameters
        ----------
        pathway : str, optional
            The pathway to execute. If None, the default pathway is used.
        overwrite : bool, optional
            If True, allows execution even if the solver is already solved.

        Raises
        ------
        RuntimeError
            If `is_solved` is True and overwrite is False.
        ValueError
            If no pathway is specified or the pathway is invalid.
        """
        if self.is_solved and not overwrite:
            raise RuntimeError("Solver is already marked as solved. Use `overwrite=True` to override.")

        pathway = pathway or self.default
        if not pathway:
            raise ValueError("No pathway specified, and no default pathway is set.")
        if pathway not in self.model._pathways:
            raise ValueError(f"Pathway '{pathway}' is not defined.")

        if not self.validate_pathway(pathway):
            raise ValueError(f"Pathway '{pathway}' validation failed.")

        steps = self.model._pathways[pathway]["processes"]
        total_steps = len(steps)

        with logging_redirect_tqdm(loggers=[self.model.logger]):
            with tqdm(total=total_steps, desc=f"Solving pathway '{pathway}'",leave=True) as pbar:
                for step_number, process in sorted(steps.items()):
                    pname,pargs,pkwargs = process['name'],process['args'],process['kwargs']
                    process = getattr(self.model, pname)
                    process(*pargs,**pkwargs)
                    self.model.logger.info(f"[SLVR] COMPLETE: {pname}")
                    pbar.set_description(f"Solving pathway '{pathway}', step: {pname}")
                    pbar.update(1)
                pbar.set_description(f"Solving pathway '{pathway}', step: DONE")
        self.is_solved = True

    def list_pathways(self) -> List[str]:
        """
        List available pathway keys.

        Returns
        -------
        List[str]
            A list of available pathway keys.
        """
        return list(self.model._pathways.keys())

    def __getitem__(self, pathway: str) -> Dict:
        """
        Retrieve pathway data.

        Parameters
        ----------
        pathway : str
            The pathway key.

        Returns
        -------
        Dict
            The pathway metadata (processes and checkers).

        Raises
        ------
        KeyError
            If the pathway does not exist.
        """
        if pathway not in self.model._pathways:
            raise KeyError(f"Pathway '{pathway}' not found.")
        return self.model._pathways[pathway]

    def get_pathway_step(self, pathway: str, index: int) -> Callable:
        """
        Retrieve a specific process step in a pathway.

        Parameters
        ----------
        pathway : str
            The pathway key.
        index : int
            The step index.

        Returns
        -------
        Callable
            The process method.

        Raises
        ------
        KeyError
            If the pathway or step index does not exist.
        """
        steps = self[pathway]["processes"]
        if index not in steps:
            raise KeyError(f"Step {index} not found in pathway '{pathway}'.")
        return getattr(self.model, steps[index])

    def get_pathway_checkers(self, pathway: str) -> List[Callable]:
        """
        Retrieve the checkers for a pathway.

        Parameters
        ----------
        pathway : str
            The pathway key.

        Returns
        -------
        List[Callable]
            List of checker methods.

        Raises
        ------
        KeyError
            If the pathway does not exist.
        """
        return [getattr(self.model, checker) for checker in self[pathway]["checkers"]]

    def validate_pathway(self, pathway: str) -> bool:
        """
        Validate the specified pathway using its checkers.

        Parameters
        ----------
        pathway : str
            The pathway key.

        Returns
        -------
        bool
            True if all checkers pass, False otherwise.
        """
        checkers = self.get_pathway_checkers(pathway)
        return all(checker(pathway) for checker in checkers)

    def find_valid_pathways(self) -> List[str]:
        """
        Identify all valid pathways.

        Returns
        -------
        List[str]
            A list of valid pathway keys.
        """
        return [pathway for pathway in self.list_pathways() if self.validate_pathway(pathway)]

    @classmethod
    def from_hdf5(cls, model: 'Model') -> 'ModelSolver':
        """
        Load solver state from an HDF5 file.

        Parameters
        ----------
        model : Model
            The model instance with an associated HDF5 handle.

        Returns
        -------
        ModelSolver
            The initialized ModelSolver instance.
        """
        handle = model.handle
        is_solved = bool(handle.attrs.get("_is_solved", False))
        default = handle.attrs.get("default", None)
        return cls(model, is_solved=is_solved, default=default)

    def to_hdf5(self):
        """
        Save solver state to an HDF5 file.
        """
        handle = self.model.handle
        handle.attrs["_is_solved"] = self.is_solved
        handle.attrs["default"] = self.default

def solver_process(path: str, step: int,args: Optional[List] = None,kwargs: Optional[Dict]=None) -> Callable:
    """
    Decorator for marking a method as part of a specific step in a pipeline.

    Parameters
    ----------
    path : str
        The pathway this process belongs to.
    step : int
        The step number within the pathway.
    args: List[Any]
        A list of arguments to pass to the method for this pipeline.
    kwargs: Dict[str, Any]
        A dictionary of keyword arguments to pass to the method for this pipeline.

    Returns
    -------
    Callable
        The decorated function.
    """

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def decorator(func: Callable) -> Callable:
        func._solver_meta = getattr(func, "_solver_meta", [])
        func._solver_meta.append({"path": path, "step": step, "type": "process","args":args, "kwargs":kwargs})
        return func

    return decorator

def solver_checker(path: str) -> Callable:
    """
    Decorator for marking a method as a validity checker for a specific pathway.

    Parameters
    ----------
    path : str
        The pathway this checker validates.

    Returns
    -------
    Callable
        The decorated function.
    """

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def decorator(func: Callable) -> Callable:
        func._solver_meta = getattr(func, "_solver_meta", [])
        func._solver_meta.append({"path": path, "type": "checker"})
        return func

    return decorator
