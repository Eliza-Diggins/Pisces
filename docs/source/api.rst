.. _api:

API
===

On this page, you can find a complete set of documentation for all of the code objects in the Pisces codebase.

Pisces Core Modules
-------------------
The **core** modules of Pisces are the modules that you (as a user) are most likely to interact with during a standard workflow.
These generally get interacted with a lot and are therefore worth being aware of when using Pisces.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    pisces.profiles
    pisces.models
    pisces.particles
    pisces.initial_conditions

Pisces Component Modules
------------------------
The **component** modules act as building blocks for the composite modules in Pisces. These generally provide things
like geometry management, profiles, etc. for models and initial conditions.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    pisces.geometry
    pisces.profiles
    pisces.dynamics

Utility Modules
---------------
These modules are just utility modules and only rarely (if ever) need to be interacted with by the user.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    pisces.utilities
    pisces.io
