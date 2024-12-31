.. _model_solvers_overview:

Model Solvers
=============

This document outlines the role of the **Model Solver** in Pisces, describes what problems it solves,
and explains its general structure.

Overview
--------

**What is a solver?**
A :py:class:`~pisces.models.solver.ModelSolver` in the Pisces framework is a component responsible for executing
an ordered series of steps (collectively called a **pathway**) to build or modify a model. For example, these steps might compute
physical quantities, initialize boundary conditions, or fill data arrays. By registering each step in
the solver’s pipeline, the solver can easily orchestrate the correct order of operations and ensure that
preconditions are met.

**What problems does it solve?**
In typical model-building workflows, you often need to:

- **Enforce ordering**: Some calculations must come before others (e.g., computing a density field before pressure).
- **Validate** the state of the model to confirm assumptions (e.g., ensuring certain parameters or arrays exist).
- **Share** a consistent mechanism for updating or extending steps.

.. note::

    The biggest benefit of this sort of structure is to reduce boiler plate code. Different "pathways" can overlap, intersect, ect.
    and therefore share code while still being distinct solution methods.

The Pisces solver approach addresses these by letting you:

1. **Declare** each step in a pipeline with a known order.
2. **Register** optional **checker** functions to confirm that the pipeline can run safely.
3. **Execute** the pipeline, ensuring each step is called in order, with consistent logging and progress tracking.

Structure of Solvers
--------------------

Pisces organizes its solver logic around these key ideas:

1. **One solver per model**:

   Each :py:class:`~pisces.models.base.Model` instance hosts exactly one solver (:py:class:`~pisces.models.solver.ModelSolver`),
   typically found at ``model._solver`` (or via an internal reference). This solver knows how to discover the model’s available
   **pathways** and manage them.

2. **Potentially many pathways**:

   A single :py:class:`~pisces.models.solver.ModelSolver` can define and handle multiple **pathways**. Each pathway is a named pipeline of steps—
   like ``"cooling_flow"`` or ``"initial_conditions"``. This approach allows you to segment or re-use logic in
   different contexts. For instance, one pathway might handle normal operation, while another sets up
   specialized boundary conditions.

3. **The solver manages and tracks each pathway**:

   Each pathway is made up of **process steps**, which are methods in your model decorated by
   :py:func:`~pisces.models.solver.solver_process`, and **checker functions**, decorated by
   :py:func:`~pisces.models.solver.solver_checker`. The solver:

   - Collects these processes in the order of their designated steps (e.g., 0, 1, 2, ...).
   - Collects any checker functions that declare themselves for that same pathway.
   - Can **validate** the pathway by invoking all checkers, ensuring each returns ``True``.
   - If valid, **executes** the pipeline steps in ascending step order.

   In code, the solver typically looks like:

   .. code-block:: python

      # Suppose `model` is a subclass of pisces.models.base.Model
      # containing a reference to a solver.

      solver = model.solver     # Access the solver
      solver("cooling_flow")    # Execute the "cooling_flow" pipeline
      # The solver runs all steps in ascending step number.

      # Validate a different pathway, e.g. "heating_flow":
      is_valid = solver.validate_pathway("heating_flow")
      if is_valid:
          solver("heating_flow")

   Every **registered** process or checker is part of the solver’s internal pipeline registry, and the solver
   orchestrates all the calls at runtime.

In summary, the solver approach centralizes the model’s stepwise logic:

1. Mark methods as solver processes (``@solver_process``) or checkers (``@solver_checker``).
2. Retrieve or run them in a cohesive pipeline using a *ModelSolver* instance (:py:class:`~pisces.models.solver.ModelSolver`).
3. Enforce ordering, optional validation, and consistent logging.

This keeps your :py:class:`~pisces.models.base.Model` class clear and modular, with all operational sequences neatly declared in
the solver’s domain.
