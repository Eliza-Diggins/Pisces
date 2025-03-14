.. _profiles-overview:

===================
Profiles in Pisces
===================

Pisces uses profiles (effectively fancy functions) as a starting point for most modeling tasks. If you're trying to
build a galaxy cluster, you'll use profiles for the dark matter density and the ICM temperature. If you're building a disk
galaxy, you'll use profiles to specify the shape of the disk and the size of the bulge.

Under the hood, all of the profiles in Pisces are contained in the :py:mod:`pisces.profiles` module. Many profiles that are
familiar from the literature are already built-in and ready to use. In some cases, you may need to define a custom profile; in
which case, you'll want to have a look at :ref:`profiles-developers`.

Overview
--------

In Pisces, profiles are all descendants of the :py:class:`~pisces.profiles.base.Profile` class which provides the vast majority
of the core functionality. It's worth reading the API documentation of that class if you're interested in the nitty-gritty, but
at their core, profiles are **just functions** with some extra structure for various purposes.

More specifically, Pisces profiles provide the following core functionality:

- Pisces profiles are **callable functions** of their variables - they act just like any other ``def`` (or ``lambda``) style
  function in python.
- Pisces profiles have **parameters** which are specified when they are initialized which allow the user to specify things
  like the shape of the profile.
- Profile classes represent **types / families of profiles** which are then instantiated to produce **functions** in that family.
- Profiles support both **numerical** and **symbolic** representations and manipulations. This allows then to occasionally
  be useful for skipping numerical steps in procedures when a symbolic solution already exists.
- Profiles have **units**.


Profiles are a fundamental component in Pisces, designed to represent mathematical models with symbolic and numerical support.
They are particularly useful for modeling quantities that depend on one or more independent variables, such as density,
temperature, or velocity distributions.

.. _prof_create_call:
Creating and Calling a Profile Instance
----------------------------------------

When you're using built-in profiles, the setup is really quite simple. You simply need to find the corresponding
:py:class:`~pisces.profiles.base.Profile` subclass and import it - then you can initialize it with the parameters of
the profile.

.. hint::

    The API documentation for each of the profiles will tell you exactly what parameters are present available. Each profile
    has a set of default parameters (:py:attr:`pisces.profiles.base.Profile.DEFAULT_PARAMETERS`) which will fill in any missing values
    that aren't specified when you create the profile instance.

Once a profile has been created using

.. code-block:: python

    >>> from pisces.profiles.density import NFWDensityProfile
    >>> density = NFWDensityProfile(**params)

you can all it just like any function:

.. code-block:: python

    >>> density(0)
    (value at r=0)

.. note::

    Profiles are not strictly 1D functions - they could be functions of many variables. In that case, you need
    to feed in each variable as a separate argument when calling the function.

.. dropdown:: Example: Plotting an NFW Profile

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

.. _prof_units:
Profile Units and Parameter Units
----------------------------------

In previous packages which inspired Pisces, model creation was done in a restrictive enough regime that there was a
"natural unit system" to use. This is not the case in Pisces and therefore has necessitated including units in all Pisces profiles.

Unit handling throughout Pisces is managed using the `unyt <https://unyt.readthedocs.io/en/stable/>`_ package which also provides
unit management for the `yt project <yt-project.org>`_. We encourage briefly checking out the documentation for ``unyt`` to
familiarize yourself with the basics of specifying unit-bearing quantities.

When creating a profile, input parameters may be **either** ``float`` or :py:class:`~unyt.unyt_quantity`. If a ``float`` is
provided, then the parameter is assumed to be using the default unit for that parameter (which is specified by the developer). If
an :py:class:`~unyt.unyt_quantity` is provided instead, then those units are propagated forward in the profile.

Likewise, the ``axes_units`` argument can be provided when initializing the profile to specify the units of the dependent axes.
Using the parameter and axes units, Pisces will automatically determine the "natural" output units for the profile and return any
outputs as a :py:class:`unyt.unyt_array` with those units.

When calling a profile, arguments without units are assumed to have the units specified in ``axes_units``. Arguments with units
are propagated consistently. You can also set the ``units`` kwarg when calling a profile to determine which output units to use.

.. dropdown:: Example: NFW With Different Units

    .. code-block:: python

        >>> from pisces.profiles.density import NFWDensityProfile
        >>>
        >>> # Create an NFW profile with the default units.
        >>> nfw_density_default_units  = NFWDensityProfile(rho_0=1e5,r_s=150) # (Assumes Msun/pc^3 and pc)
        >>> print(nfw_density_default_units.output_units)
        Msun/pc**3
        >>> # Create an NFW profile in cgs.
        >>> from unyt import unyt_quantity as uq
        >>> nfw_density_default_units  = NFWDensityProfile(rho_0=uq(1,'g/cm**3'),r_s=uq(10,'m'))
        >>> print(nfw_density_default_units.output_units)
        g/cm**3

.. hint::

    You can check what units are being used when you have a profile by interacting with the :py:attr:`~pisces.profiles.base.Profile.axes_units`,
    :py:attr:`~pisces.profiles.base.Profile.parameters`, and :py:attr:`~pisces.profiles.base.Profile.output_units` attributes.

Working With Profiles Symbolically
----------------------------------

Underlying the Pisces profile system is the `sympy <https://www.sympy.org/en/index.html>`_ package, which provides support
for computer algebra in python. Under the hood, each :py:class:`~pisces.profiles.Profile` class has a set of symbolic axes
(:py:attr:`~pisces.profiles.Profile.SYMBAXES`) and a set of symbolic parameters (:py:attr:`~pisces.profiles.Profile.SYMBPARAMS`).
When the class is created (when you import ``pisces``), Sympy will automatically build a **symbolic** version of the profile.

Later on, when you initialize an instance of the profile, the parameters get substituted into the symbolic expression and
the entire expression is converted to an efficient (numpy based) callable function. This is extremely versatile because users
have access to both **symbolic manipulations** and **numerical manipulations** of the profile. In many cases, some (potentially
expensive) numerical operations can be skipped over because an analytic solution is already known to exist for a particular profile. Access
to the symbolic expressions allows these sorts of optimizations to be easily incorporated into the code base.

In each profile class, the :py:attr:`~pisces.profiles.base.Profile.profile_expression` contains the symbolic representation of
the profile:

.. code-block:: python

    >>> from pisces.profiles.density import NFWDensityProfile
    >>> print(NFWDensityProfile.profile_expression)
    r_s*rho_0/(r*(r/r_s + 1)**2)

Each of the symbols in this expression is either a parameter symbol or a variable symbol as described in the preceding paragraph.

Once the profile is initialized, you can access the "simplified" expression with parameters substituted in using :py:attr:`~pisces.profiles.base.symbolic_expression`.

    >>> from pisces.profiles.density import NFWDensityProfile
    >>> prof = NFWDensityProfile()
    >>> print(prof.symbolic_expression)
    1.0/(r*(1.0*r + 1)**2)

Derived Expressions
+++++++++++++++++++

One of the most useful features of Pisces profiles is the ability to derive new symbolic representations from expressions. We call
these **derived expressions** and they can be created / manipulated in a variety of ways.

In many cases, profiles (particularly well known profiles) have derived expressions which built into the Pisces infrastructure. These
are called **class-level derived expressions** because they are provided as part of the profile class and are written by a developer.
To see a list of these derived expressions, you can simply call :py:meth:`~pisces.profiles.base.Profile.list_class_expressions`.

.. code-block:: python

    >>> prof = NFWDensityProfile()
    >>> prof.list_class_expressions()
    ['spherical_potential', 'spherical_mass', 'derivative']

.. note::

    Not all profiles have any derived expressions at the class level - others may have many depending on
    the relevance of the profile.

In the example above, there are 3 class expressions representing different properties of the NFW profile. To access a derived attribute
at the class level, simply use :py:meth:`~pisces.profiles.base.Profile.get_class_expression`

.. code-block:: python

    >>> prof = NFWDensityProfile()
    >>> prof.get_class_expression('spherical_mass')
    4*pi*r_s**3*rho_0*(-r/(r + r_s) + log(r/r_s + 1))

In addition to **class-level derived expressions**, there are also **instance-level derived expressions** which provide
effectively the same functionality as class level expressions; however, with the specific parameter values already substituted
into the expression. These can be accessed and manipulated via :py:meth:`pisces.profiles.base.Profile.get_expression`,
:py:meth:`pisces.profiles.base.Profile.set_expression`, and :py:meth:`pisces.profiles.base.Profile.has_expression`.

Additionally, the **instance-level derived expressions** can be converted to **numerical** expressions which can then
be used for computation:

.. code-block:: python

    >>> prof = NFWDensityProfile()
    >>> pot = prof.get_numeric_expression('spherical_mass')


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
