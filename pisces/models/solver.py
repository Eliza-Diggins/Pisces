"""
Solver classes for Pisces models

Overview
--------

A ModelSolver in the Pisces framework is a component responsible for executing an ordered series of steps (collectively
called a pathway) to build or modify a model. For example, these steps might compute physical quantities, initialize
boundary conditions, or fill data arrays. By registering each step in the solver’s pipeline, the solver can easily
orchestrate the correct order of operations and ensure that preconditions are met.

For detailed documentation on how the solver infrastructure works, reference :ref:`model_solvers_overview`.

"""
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pisces.utilities.config import pisces_params
from pisces.utilities.logging import devlog, mylog

if TYPE_CHECKING:
    from pisces.models.base import Model


# noinspection PyUnresolvedReferences,PyProtectedMember
class ModelSolver:
    """
    Model solver base class.

    The :py:class:`ModelSolver` class is attached to every :py:class:`~pisces.models.base.Model` instance and is responsible for
    the following tasks:

    - **Identifying** the available solver pipelines attached to the :py:class:`~pisces.models.base.Model` class.
    - **Validating** that pipelines are contiguous and can be executed.
    - **Check** whether a specific pipeline is valid given a set of conditions.
    - **Run** the pipeline to solve the :py:class:`~pisces.models.base.Model` instance.

    .. note::

        The idea behind this class is to abstract away the logic for locating / managing pipelines away from the
        model itself to make the code base more readable. Instead, the :py:class:`ModelSolver` class is in charge of the
        entire execution segment of the model.

    Parameters
    ----------
    model : :py:class:`~pisces.models.base.Model`
        The model instance this solver is associated with.
    is_solved : bool, optional
        Tracks whether the solver has been initialized or solved (default: False).
    default : str, optional
        Default pathway to execute if none is specified (default: None).

    Attributes
    ----------
    model : :py:class:`~pisces.models.base.Model`
        The associated model instance.
    is_solved : bool
        Tracks whether the solver has been initialized or solved.
    default : str
        Default pathway to execute.
    """
    def __init__(self, model: 'Model', is_solved: bool = False, default: Optional[str] = None):
        """
        Initialize the :py:class:`~pisces.models.solver.ModelSolver` class.

        Parameters
        ----------
        model : :py:class:`~pisces.models.base.Model`
            The associated model instance.
        is_solved : bool, optional
            Tracks whether the solver has been initialized or solved. By default, this is ``False``.
        default : str, optional
            Default pathway to execute. By default this is ``None``, meaning that the user must
            specify the pipeline at execution.
        """
        self.model: 'Model' = model
        self.is_solved: bool = is_solved
        self.default: Optional[str] = default

    def __call__(self, pathway: Optional[str] = None, overwrite: bool = False):
        """
        Execute a particular pathway to solve the associated model.

        Calling the :py:class:`ModelSolver` instance will cause it to attempt to solve the :py:class:`~pisces.models.base.Model` instance
        using the provided ``pathway``.

        Parameters
        ----------
        pathway : str, optional
            The pathway to execute. If None, the default pathway is used.
        overwrite : bool, optional
            If ``True``, allows execution even if the solver is already solved.

        Raises
        ------
        RuntimeError
            If `is_solved` is True and overwrite is False.
        ValueError
            If no pathway is specified or the pathway is invalid.
        """
        # VALIDATION: check if the solver is already solved and proceed accordingly.
        if self.is_solved and not overwrite:
            raise RuntimeError("Solver is already marked as solved. Use `overwrite=True` to override.")

        # DETERMINE the relevant pathway or utilize the default.
        pathway = pathway or self.default
        if not pathway:
            raise ValueError("No pathway specified, and no default pathway is set.")
        if pathway not in self.model._PATHWAYS:
            raise ValueError(f"Pathway '{pathway}' is not defined.")

        if not self.validate_pathway(pathway):
            raise ValueError(f"Pathway '{pathway}' validation failed.")

        # SETUP the runtime: determine the number of steps and prepare to
        # execute.
        steps = self.model._PATHWAYS[pathway]["processes"]
        total_steps = len(steps)

        with logging_redirect_tqdm(loggers=[self.model.logger,devlog,mylog]):
            with tqdm(total=total_steps,
                      desc=f"[EXEC]",
                      leave=True,
                      disable=pisces_params['system.preferences.disable_progress_bars']) as pbar:
                for step_number, process in sorted(steps.items()):
                    # Determine the process and run it.
                    pname,pargs,pkwargs = process['name'],process['args'],process['kwargs']
                    process = getattr(self.model, pname)
                    self.model.logger.info(f"[EXEC] \t(%s/%s) START: `%s`.",step_number + 1, total_steps,pname)
                    try:
                        process(*pargs,**pkwargs)
                    except Exception as e:
                        self.model.logger.error(f"[EXEC] \t(%s/%s) FAILED: `%s`.", step_number + 1, total_steps,
                                               pname)
                        pbar.set_description(f"[EXEC] STATUS: FAILED")
                        raise e
                    self.model.logger.info(f"[EXEC] \t(%s/%s) COMPLETE: `%s`.", step_number + 1, total_steps,
                                               pname)
                    pbar.set_description(f"[EXEC] STEP: {pname}")
                    pbar.update(1)
                pbar.set_description(f"[EXEC] STATUS: DONE")
        self.is_solved = True

    def list_pathways(self) -> List[str]:
        """
        List available pathway keys.

        Returns
        -------
        List[str]
            A list of available pathway keys.
        """
        return list(self.model._PATHWAYS.keys())

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
        if pathway not in self.model._PATHWAYS:
            raise KeyError(f"Pathway '{pathway}' not found.")
        return self.model._PATHWAYS[pathway]

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

def solver_process(path: str,
                   step: int,
                   args: Optional[List] = None,
                   kwargs: Optional[Dict]=None) -> Callable:
    r"""
    Decorate a model method to register it as a process step within a solver pathway.

    The :py:func:`solver_process` decorator associates a method in a :py:class:`~pisces.models.base.Model` subclass
    with a particular solution pathway. When the solver is instructed to execute that pathway, the decorated method will be
    invoked in the correct order (according to its specified ``step``) with the provided positional
    and keyword arguments (``args`` and ``kwargs``).

    Parameters
    ----------
    path : str
        The solver pathway name under which this process is registered.

        The solver pathway should be a unique string identifier for the particular pathway being registered to. If two pathways
        share a name, errors will be raised.

        .. hint::

            It's often useful to name these after some sort of starting condition and the relevant coordinate systems
            for which the pathway is valid.

    step : int
        The order (step number) in which this process should be executed within the pathway.
        Smaller values of ``step`` are executed earlier, and so on.

        .. warning::

            Duplicate steps in the same pathway are not permitted.

    args : list, optional
        A list of arguments to pass to the decorated function when the solver executes
        this process step. Defaults to ``None``, in which case no additional arguments are passed.
    kwargs : dict, optional
        A dictionary of keyword arguments to pass to the decorated function. Defaults
        to ``None``, in which case no additional keyword arguments are passed.

    Returns
    -------
    callable
        The decorated function, annotated with metadata that the solver can use to build
        an ordered pipeline of steps.

    Notes
    -----
    - The solver stores the metadata for each registered process (pathway, step, and arguments)
      under a private attribute (``._solver_meta``) on the function object.
    - When the solver runs, it retrieves these metadata records to construct the execution
      pipeline for each pathway.
    - You can register multiple functions under the same pathway, each with a different ``step``
      value, forming a sequential procedure. You can also register a single function more than once at different
      steps in a single pathway, allowing for cyclic execution.

    Examples
    --------
    .. code-block:: python

       from pisces.models.solver import solver_process

       class MyModel(Model):
           @solver_process(path="cooling_flow", step=0, args=["some_data"], kwargs={"flag": True})
           def compute_density(self, data, flag=False):
               # Implementation for density computations
               pass

           @solver_process(path="cooling_flow", step=1)
           def compute_pressure(self):
               # Implementation for pressure computations
               pass

    In the example above:

    - The ``compute_density`` method is registered as step 0 in the ``cooling_flow`` pathway,
      with ``some_data`` and ``flag=True`` passed at runtime.
    - The ``compute_pressure`` method is registered as step 1 in the same pathway, running
      immediately after ``compute_density``.

    When this model's solver is told to run the ``cooling_flow`` pathway, the solver looks
    up these metadata records, calls ``compute_density("some_data", flag=True)``, and then
    calls ``compute_pressure()``.
    """
    # noinspection PyProtectedMember,PyUnresolvedReferences
    def decorator(func: Callable) -> Callable:
        func._solver_meta = getattr(func, "_solver_meta", [])
        func._solver_meta.append({"path": path, "step": step, "type": "process","args":args, "kwargs":kwargs})
        return func

    return decorator

def serial_solver_processes(entries: List[Tuple[str, int, List[Any], Dict[str, Any]]]) -> Callable:
    """
    Decorator that applies multiple :py:func:`solver_process` calls (one for each entry in ``entries``)
    to the same function in series.

    Each tuple in ``entries`` must be of the form: ``(path, step, args_list, kwargs_dict)``

    The decorator will, in order:
      - Wrap the target function with :py:func:`solver_process`.
      - Accumulate the ``_solver_meta`` metadata so that your solver can run these steps.

    Parameters
    ----------
    entries : List[Tuple[str, int, List[Any], Dict[str, Any]]]
        A list of process descriptions. Each element is a 4-tuple: ``(path, step, args_list, kwargs_dict)``.

    Returns
    -------
    Callable
        A decorator that, when applied to a function, returns a function with
        aggregated solver-process metadata for each of the given entries.

    Examples
    --------
    .. code-block:: python

        from mymodule import solver_process, serial_solver_processes

        # Suppose these are the pipeline entries for 'cooling_flow' pathway:
        process_descriptions = [
            ("cooling_flow", 0, ["some_data"], {"flag": True}),
            ("cooling_flow", 1, [], {})
        ]

        @serial_solver_processes(process_descriptions)
        def compute_flows(*args, **kwargs):
            # Implementation of your solver logic.
            pass

        # Now `compute_flows._solver_meta` includes two solver-process records:
        # [
        #   {'path': 'cooling_flow', 'step': 0, 'type': 'process',
        #    'args': ['some_data'], 'kwargs': {'flag': True}},
        #   {'path': 'cooling_flow', 'step': 1, 'type': 'process',
        #    'args': [], 'kwargs': {}}
        # ]

    """

    def decorator(func: Callable) -> Callable:
        # Start with the original function object,
        # then iteratively decorate it with solver_process for each entry.
        for path, step, arg_list, kw_dict in entries:
            # Repeatedly apply solver_process to func
            func = solver_process(path, step, arg_list, kw_dict)(func)

        return func

    return decorator

def solver_checker(path: str) -> Callable:
    r"""
    Decorate a model method to register it as a "checker" for a solver pathway.

    A *checker* is a validation function for a particular pathway. When the solver
    is instructed to validate that pathway (for instance, via
    :py:meth:`~pisces.models.solver.ModelSolver.validate_pathway`), any decorated
    checker function is retrieved and called with the pathway name as its only
    argument. The checker should return ``True`` if the pathway is valid (or some
    portion thereof) and ``False`` otherwise.

    Parameters
    ----------
    path : str
        The name of the solver pathway for which this checker is registered.

        .. tip::

            Often, the name of the pathway corresponds to a multi-step procedure in your
            model (e.g. "cooling_flow"). By registering a checker, you can ensure that
            certain preconditions or assumptions about the data hold before (or after)
            the solver runs that pathway.

    Returns
    -------
    Callable
        The decorator function that, when applied to a method, annotates it with
        the solver metadata so the solver can retrieve it as part of the validation
        routine.

    Notes
    -----
    - All checker functions for a pathway are stored in a private list
      on the function object itself, under the attribute
      ``._solver_meta``. The solver manager (usually
      :py:class:`~pisces.models.solver.ModelSolver`) inspects these records
      and calls each checker in turn to see if the entire pathway is valid.
    - A checker must accept a single argument (the pathway name) and return
      a boolean. Typically, you’ll fetch or read from the model or relevant
      fields to confirm the pathway is consistent.
    - You can have multiple checkers for the same pathway. The solver considers
      a pathway *valid* if **all** checkers for it return ``True``.

    Examples
    --------
    .. code-block:: python

       from pisces.models.solver import solver_checker, solver_process

       class MyModel(Model):

           @solver_process(path="cooling_flow", step=0)
           def do_stuff(self):
               # Implementation that modifies the model
               pass

           @solver_checker(path="cooling_flow")
           def ensure_sanity(self, pathway):
               # Return True if the model is in a valid state for this pathway
               # e.g., check if a relevant field is already computed or a parameter is set
               return hasattr(self, "some_critical_value")

    Here, ``ensure_sanity`` will be invoked whenever the solver tries to validate
    the "cooling_flow" pathway. If it returns ``False``, the solver can raise an error
    or skip execution, thereby preventing a possibly inconsistent run.
    """
    # noinspection PyProtectedMember,PyUnresolvedReferences
    def decorator(func: Callable) -> Callable:
        func._solver_meta = getattr(func, "_solver_meta", [])
        # We store the minimal metadata needed to identify this function
        # as a checker for a specific pathway.
        func._solver_meta.append({"path": path, "type": "checker"})
        return func

    return decorator

def serial_solver_checkers(paths: List[str]) -> Callable:
    r"""
    Decorator that applies multiple :py:func:`solver_checker` calls (one for each
    specified pathway in ``paths``) onto the same function in series.

    This is useful when a single validation function needs to serve as a checker
    for multiple pathways. The function is effectively *declared* as a checker
    for every pathway listed in ``paths``.

    Parameters
    ----------
    paths : List[str]
        A list of solver pathway names for which this function should serve
        as a checker.

    Returns
    -------
    Callable
        A decorator that, when applied to a function, annotates it with multiple
        checker records, one for each pathway in ``paths``.

    Examples
    --------
    .. code-block:: python

        from pisces.models.solver import solver_checker, serial_solver_checkers

        @serial_solver_checkers(["cooling_flow", "heating_flow"])
        def ensure_universal_sanity(self, pathway):
            # This checker applies to both 'cooling_flow' AND 'heating_flow'
            # Return True/False based on conditions in self
            required_attr = "some_universal_flag"
            return getattr(self, required_attr, False)

        # `ensure_universal_sanity._solver_meta` might look like this:
        # [
        #   {'path': 'cooling_flow', 'type': 'checker'},
        #   {'path': 'heating_flow', 'type': 'checker'}
        # ]

        # The solver will call `ensure_universal_sanity('cooling_flow')`
        # when validating the 'cooling_flow' pathway, and
        # `ensure_universal_sanity('heating_flow')` for 'heating_flow'.
    """
    def decorator(func: Callable) -> Callable:
        # Start with the original function object,
        # then iteratively decorate it with solver_checker for each path.
        for path in paths:
            func = solver_checker(path)(func)
        return func

    return decorator
