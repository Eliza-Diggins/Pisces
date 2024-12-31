.. _profiles-overview:
Profiles in Pisces
===================

Profiles are a fundamental component in Pisces, designed to represent mathematical models with symbolic and numerical support.
They are particularly useful for modeling quantities that depend on one or more independent variables, such as density,
temperature, or velocity distributions.

This guide provides an overview of the :py:class:`~pisces.profiles.base.Profile` base class, its key attributes and methods,
and how to create your own profiles.

At their core, profiles in Pisces are mathematical functions parameterized by:

- **Independent variables** (:py:attr:`~pisces.profiles.base.Profile.AXES`): Variables that the profile depends on.
- **Parameters** (:py:attr:`~pisces.profiles.base.Profile.DEFAULT_PARAMETERS`): Values that define the behavior or shape of the profile.
- **Units** (:py:attr:`~pisces.profiles.base.Profile.DEFAULT_UNITS`): The dimensionality of the profile's output.

Each :py:class:`~pisces.profiles.base.Profile` has a base function which incorporates these axes and parameters, and each
instance of the profile represents a specific choice of the parameters.

Profiles support both symbolic manipulation (via `Sympy <https://www.sympy.org>`_) and numerical evaluation (via ``numpy``),
making them versatile for analytical and computational tasks.

Initializing and Calling Profiles
---------------------------------

At the surface, :py:class:`~pisces.profiles.base.Profile` classes are easy to work with. To initialize one, you simply need
to feed it the relevant parameters (:py:attr:`~pisces.profiles.base.Profile.DEFAULT_PARAMETERS`) as ``kwargs``. For example, we can
create and plot a :py:class:`~pisces.profiles.density.NFWDensityProfile` very simply:

.. dropdown:: Example

    .. plot::
        :include-source:

        >>> from pisces.profiles.density import NFWDensityProfile
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> nfw_density = NFWDensityProfile(rho_0=1e5,r_s=150)
        >>> radii = np.geomspace(1e-2,1e4,1000)
        >>>
        >>> # Create the plot.
        >>> plt.loglog(radii,nfw_density(radii),'k-')
        >>> plt.ylabel(r"Density, Msun/kpc^3")
        >>> plt.xlabel(r"Radius, kpc")
        >>> plt.show()

You can easily access the parameters (:py:attr:`~pisces.profiles.base.Profile.parameters`) and the units (:py:attr:`~pisces.profiles.base.Profile.units`)
of your profile instance at any time.

.. tip::

    You can always set the output units of a profile by specifying ``units=...`` when initializing it. The units must
    be consistent the with class default (:py:attr:`~pisces.profiles.base.Profile.DEFAULT_UNITS`), but other than that constraint,
    they may be chosen to fit your need.

Symbolic and Numeric Expressions
--------------------------------

Each profile class supports both symbolic and numerical representations of its underlying function as well as symbolic and numerical
representations of special attributes like derivatives, integrals, and other properties. To access the underlying function, you can use
either :py:attr:`~pisces.profiles.base.Profile.symbolic_expression` to access the instance-level version (with parameters substituted in) or
:py:attr:`~pisces.profiles.base.Profile.class_symbolic_expression` to access the class-level version (without parameter substitution).

Other symbolic expressions related to the base function may exist for specific profiles. They are always registered in one of two
places / categories:

- **Class-Level Expressions**: These are intrinsic to the **class**.
- **Instance-Level Expressions**: These are intrinsic to each **instance**

The instance level expressions can also be converted to **numerical expressions** which are then well suited to computational
tasks.

.. note::

    You cannot make class level expressions numerical because they still contain symbols for parameters. You can create an
    instance-level version of any class-level expression and then create a numerical version of that. See the sections below
    for details.

Class Level Expressions
+++++++++++++++++++++++

Class-level expressions are derived symbolic attributes shared across all instances of a profile class. These expressions
are often used to represent analytical properties like derivatives or asymptotic behaviors. The following functions provide
the user with interaction capabilities

- **Define**: Use the :py:meth:`~pisces.profiles.base.Profile.set_class_expression` method to register a symbolic expression.
- **Access**: Use the :py:meth:`~pisces.profiles.base.Profile.get_class_expression` method to retrieve a registered symbolic expression.

Class level expressions are functions of the symbolic axes (:py:attr:`~pisces.profiles.base.Profile.SYMBAXES`) and the
symbolic parameters (:py:attr:`~pisces.profiles.base.Profile.SYMBPARAMS`).

Instance Level Expressions
++++++++++++++++++++++++++

Instance-level expressions are specific to a particular instance of a profile and can override
or extend class-level definitions. These expressions depend on the instance's parameter values and are therefore
only functions of the symbolic axes (:py:attr:`~pisces.profiles.base.Profile.SYMBAXES`).

Every **class-level** expression can be converted to a **instance-level** expression. This conversion is (by default) done
automatically when fetching an instance level attribute. Just like the class level attributes, you can access the instance level
attributes as

- **Define**: Use the :py:meth:`~pisces.profiles.base.Profile.set_expression` method to register a symbolic expression.
- **Access**: Use the :py:meth:`~pisces.profiles.base.Profile.get_expression` method to retrieve a registered expression.

Numerical Expressions
+++++++++++++++++++++

Just as the :py:class:`~pisces.profiles.base.Profile` class keeps track of symbolic expressions, it can also keep track
of numerical expressions. For every **instance-level** expression, there is also a numerical equivalent accessed using
:py:meth:`~pisces.profiles.base.Profile.get_numeric_expression` which will pull the numerical expression from the class's repository.
If the numerical version of an expression has not been used before, it is "lambdified" from the symbolic expression to produce
the callable function. These always take the :py:attr:`~pisces.profiles.base.Profile.AXES` as inputs (separate slots).

Serialization and IO Procedures
-------------------------------

Profiles in Pisces support serialization and deserialization using the HDF5 file format, allowing users to save profile
instances and later reload them. This functionality ensures that profiles can be stored persistently or shared across different environments.

Saving Profiles
+++++++++++++++

To save a profile instance, use the :py:meth:`~pisces.profiles.base.Profile.to_hdf5` method. This method writes the profile's
class name, axes, units, and parameters to a specified HDF5 file or group.

.. code-block:: python

    import h5py
    from pisces.profiles.density import NFWDensityProfile

    # Create a profile instance
    profile = NFWDensityProfile(rho_0=1e5, r_s=150)

    # Save the profile to an HDF5 file
    with h5py.File('profile.h5', 'w') as f:
        profile.to_hdf5(f, group_name='nfw_profile')

The `group_name` parameter specifies the HDF5 group under which the profile data will be saved. If a group with the same
name already exists and `overwrite=False` (default), the method will raise an error.

.. note::

    The method stores only the profile's defining attributes (axes, parameters, and units) and not any custom instance-level
    expressions. If additional data is associated with your profile, you must manage its storage separately.

Loading Profiles
++++++++++++++++

To load a profile from an HDF5 file, use the :py:meth:`~pisces.profiles.base.Profile.from_hdf5` method. This method
reconstructs a profile instance using the saved attributes.

.. code-block:: python

    # Load the profile from the HDF5 file
    with h5py.File('profile.h5', 'r') as f:
        loaded_profile = NFWDensityProfile.from_hdf5(f, group_name='nfw_profile')

The method automatically identifies the correct profile class based on the saved class name and initializes it with the
stored parameters and units.

.. tip::

    You can inspect the contents of an HDF5 file (e.g., using `h5py.File.keys()`) to view the available groups and their
    structure before loading profiles.

Limitations
+++++++++++

- **Custom Attributes**: Only the attributes defined in the profile class are saved. Custom attributes or instance-level
  symbolic expressions must be managed separately.
- **Compatibility**: Ensure that the same version of Pisces is used when saving and loading profiles, as changes in class
  definitions may lead to incompatibilities.

Use Cases
+++++++++

- **Persistent Storage**: Save profiles to files for later reuse, avoiding the need to redefine them.
- **Portability**: Share profiles between users or systems using a standardized file format.
- **Batch Processing**: Save multiple profiles in a single HDF5 file, grouped by name, for streamlined data management.