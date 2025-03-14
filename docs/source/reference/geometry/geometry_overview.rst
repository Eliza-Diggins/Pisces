.. _geometry_overview:

============================
Coordinate Systems in Pisces
============================

Pisces is a versatile model-building software designed for astrophysical applications across a wide range of scales
and geometries. It supports modeling scenarios from galaxy clusters to stellar structures, as well as more
intricate astrophysical environments.

One of Pisces' core strengths is its ability to handle complex physical processes in diverse and flexible
coordinate systems. This capability allows users to work seamlessly with different geometries, making
it suitable for a variety of astrophysical contexts.

On this page—and the linked sub-pages—you’ll find a comprehensive overview of the theory and implementation behind
Pisces’ coordinate system framework, along with guidance on how to leverage it for your own modeling needs.

Overview
--------

Coordinate systems and geometric computations are a complicated aspect of Pisces and (in some ways) are also the most
critical part of the underlying infrastructure. Everything geometry related is managed through the :py:mod:`pisces.geometry` module,
which features sub-modules for the following

- :py:mod:`~pisces.geometry.coordinate_systems`: Provides classes representing all of the different coordinate systems you
  can use in Pisces models as well as support for differential operations like the gradient, divergence, and laplacian.
- :py:mod:`~pisces.geometry.handler`: Provides classes which merge coordinate systems with **symmetry** to reduce the complexity
  of certain operations.

Together, these two modules are used in pretty much every component of the Pisces ecosystem.



Theory
------

Arbitrary coordinate systems are a complex subject unto themselves and require a relatively sophisticated understanding
of the relevant mathematics to fully digest. For readers who are not familiar with the theory of orthogonal coordinate systems,
including the use of Lame coefficients and the construction of differential operators, we suggest reading the documentation
provided on :ref:`geometry_theory` to get an overview of the relevant background.

Pisces Coordinate Systems
-------------------------

All of the coordinate systems in Pisces are located in the :py:mod:`~pisces.geometry.coordinate_systems` module and can be
imported directly from there. In general, each of these classes works very similarly, but they are very deep, complex classes and
so some familiarity with the capabilities of these classes is worthwhile. In this section, we will cover the various things that you
can do with a coordinate system in Pisces.

Creating a Coordinate System Instance
'''''''''''''''''''''''''''''''''''''

Depending on the coordinate system you're working with, there may or may not be parameters needed to specify the coordinate system
when it in initialized. You should consult the documentation to see what parameters are necessary. For example, the spherical coordinate
system has no parameters and may be initialized as

.. code-block:: python

    from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
    spherical_coords = SphericalCoordinateSystem()

If you want to initialize a :py:class:`~pisces.geometry.coordinate_systems.OblateHomoeoidalCoordinateSystem`, then you'll need to specify
the eccentricity of the coordinate system:

.. code-block:: python

    from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
    ellipsoidal_coords = OblateHomoeoidalCoordinateSystem(ecc=0.5)

The Symbolic Coordinate System Interface
''''''''''''''''''''''''''''''''''''''''

Pisces coordinate systems utilize a dual approach to balance complexity and numerical efficiency. We refer to these two
sides of the coordinate systems as the two "interfaces":

1. The **symbolic interface** is managed by the `Sympy <https://www.sympy.org>`_ symbolic mathematics package and provides
   access to various analytically derived attributes of the coordinate systems including Lame Coefficients and other derived
   components of the differential operators. The symbolic interface also handles symmetry management.
2. The **numerical interface** is managed via ``numpy`` and allows the user to rapidly access array-vectorized functions for
   various processes. Most importantly, these functions can be used for taking differential operations, converting between coordinate systems, etc.

In many cases, there are both symbolic and numerical equivalents of particular attributes in a coordinate system. For example,
the Lame coefficients of the coordinate system are accessible to the user through the :py:meth:`~pisces.geometry.base.CoordinateSystem.get_lame_function` in
the numerical interface and through the :py:meth:`~pisces.geometry.base.CoordinateSystem.get_lame_symbolic` in the symbolic
interface.

In this section, we'll focus on the **symbolic interface** before turning our attention to the numeric interface in the
remainder of the document.

Accessing Symbolic Attributes
+++++++++++++++++++++++++++++

Foundationally, every :py:class:`~pisces.geometry.base.CoordinateSystem` class has two sets of sympy symbols from which all of
the symbolic expression are constructed:

- :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBAXES`: provides the symbols for each of the coordinate variables.
- :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBPARAMS`: provides symbolic access to the parameters of the coordinate system.

From these, a variety of attributes may be derived. Which attributes are derived depends on the coordinate system in question and its properties. All of
these are accessed using the :py:meth:`~pisces.geometry.base.CoordinateSystem.get_derived_attribute_symbolic` and
:py:meth:`~pisces.geometry.base.CoordinateSystem.set_derived_attribute_symbolic`.

.. hint::

    Consult the documentation for your coordinate system to figure out what attributes are available for the specific
    coordinate system of interest.

There are 3 symbolic expressions which are **always available** in every coordinate system:

- **Lame Coefficients**: See :py:meth:`~pisces.geometry.base.CoordinateSystem.get_lame_symbolic`
- **D-Term**: See :py:meth:`~pisces.geometry.base.CoordinateSystem.get_symbolic_D_term`
- **L-Term**: See :py:meth:`~pisces.geometry.base.CoordinateSystem.get_symbolic_L_term`.

Performing Symbolic Operations
++++++++++++++++++++++++++++++

You can use ``sympy`` to do any number of things with the symbolic expression available in the **symbolic interface**; however,
a couple of central cases are built into the :py:class:`~pisces.geometry.base.CoordinateSystem` class:

- **Gradient**: :py:meth:`~pisces.geometry.base.CoordinateSystem.analytical_gradient`
- **Divergence**: :py:meth:`~pisces.geometry.base.CoordinateSystem.analytical_divergence`
- **Laplacian**: :py:meth:`~pisces.geometry.base.CoordinateSystem.analytical_laplacian`

Specific coordinate systems may provide special symbolic methods that can be used as well.

The Numerical Coordinate System Interface
''''''''''''''''''''''''''''''''''''''''''

While the symbolic interface is largely used in the backend to make certain procedures more efficient, the numeric interface provides
the core of the user-side ability of the :py:class:`~pisces.geometry.base.CoordinateSystem` class.

The **numerical interface** in Pisces provides a fast and efficient way to perform operations with coordinate systems using `numpy`.
This interface is the primary means for users to interact with coordinate systems for numerical computations.
Below are the key aspects of the numerical interface:

Accessing Numerical Expressions
+++++++++++++++++++++++++++++++

Various important quantities for coordinate systems are present in both the symbolic and numerical interface. In some cases,
these attributes are converted from symbolic to numeric form when the user instantiates an instance of the class; in others, they
are converted to numerical form on the fly. In every coordinate system class, the following are accessible:

- **Lame Coefficients**:

  The Lame coefficients are accessible via the numerical interface to account for scale factors in differential operations. The numerical
  forms can be accessed via the :py:meth:`~pisces.geometry.base.CoordinateSystem.get_lame_function`, which will return the
  numerical form of the Lame Coefficients for a particular axis. These are efficiently converted from the symbolic form on instantiation.

- **Jacobian**:

  The Jacobian is accessed via :py:meth:`~pisces.geometry.base.CoordinateSystem.jacobian`.

- **Derived Attributes**:

  Any symbolic attributes derived during the symbolic setup can be accessed as lambdified numerical functions. These can be
  fetched using the :py:meth:`~pisces.geometry.base.CoordinateSystem.get_derived_attribute_function` method. The user can also
  set new numerical attributes using the :py:meth:`~pisces.geometry.base.CoordinateSystem.set_derived_attribute_function` method.

  .. note::

        If a particular attribute is present as a symbolic attribute and the user asks for it numerically, it will be converted;
        however, this can be expensive and should be done with some degree of care.

Converting Between Coordinate Systems
+++++++++++++++++++++++++++++++++++++

Coordinate transformations are a key feature of Pisces coordinate systems. Conversion between a custom coordinate system
and Cartesian coordinates (and vice versa) is straightforward and managed with :py:meth:`~pisces.geometry.base.CoordinateSystem.to_cartesian`
and :py:meth:`~pisces.geometry.base.CoordinateSystem.from_cartesian`.

To convert between other coordinate system classes, we first convert to cartesian and then adopt the other class's method
to convert back from cartesian to the target coordinate system. This is all managed behind the scenes via :py:meth:`~pisces.geometry.base.CoordinateSystem.convert_to`.


Differential Operations
+++++++++++++++++++++++++++++++

Like the symbolic interface, the numerical interface also provides the ability to compute differential operations efficiently:

- **Gradient**: :py:meth:`~pisces.geometry.base.CoordinateSystem.compute_gradient`
- **Divergence**: :py:meth:`~pisces.geometry.base.CoordinateSystem.compute_divergence`
- **Laplacian**: :py:meth:`~pisces.geometry.base.CoordinateSystem.compute_laplacian`

Specific coordinate systems may provide special symbolic methods that can be used as well.

Geometry Handlers and Symmetry
------------------------------

Geometry handlers in Pisces provide an essential interface for managing and simplifying computations in coordinate systems
under symmetry constraints. These utilities are particularly valuable in handling the complex interplay between symmetric
and free axes, enabling efficient operations such as gradients, divergences, and Laplacians.

Most importantly, geometry handlers can help you avoid having to provide "dummy" coordinates that you don't care about because
of symmetry.

The core class for geometry handling is :py:class:`~pisces.geometry.handler.GeometryHandler`, which takes two arguments:

- A :py:class:`~pisces.geometry.base.CoordinateSystem` instance and
- A list of "free-axes", corresponding to the axes that still matter after considering symmetry.

The :py:class:`~pisces.geometry.handler.GeometryHandler` will then manage all of the following tasks

1. **Symmetry Management**:

   - Differentiates between free axes (axes relevant for computations) and symmetric axes (axes invariant under symmetry).
   - Automatically handles default values for symmetric axes during computations.

2. **Dynamic Subclassing**:

   - Automatically selects the appropriate specialized handler subclass based on the coordinate system's characteristics.

3. **Coordinate Handling**:

   - Coerces partial coordinates into full dimensions, ensuring compatibility with the underlying coordinate system.
   - Manages default values for symmetric axes, enabling users to specify only the relevant free axes.

4. **Dependency Analysis**:

   - Identifies dependencies for differential operations (gradient, divergence, Laplacian) based on the symbolic attributes of the coordinate system.
   - Determines which axes or terms contribute to these operations, optimizing computations.

5. **Numerical and Symbolic Integration**:

   - Combines the symbolic and numerical attributes of the coordinate system for accurate and efficient computation of derived quantities.

Coordinate Management
+++++++++++++++++++++++++

The :py:class:`~pisces.geometry.base.GeometryHandler` can be instantiated dynamically based on the
associated coordinate system. If the coordinate system specifies a specialized handler (via ``_handler_class_name``),
the appropriate subclass is instantiated automatically.

The resulting :py:class:`~pisces.geometry.handler.GeometryHandler` mimics many of the methods present in the underlying
coordinate system, but takes a different structure for specifying coordinates. Unlike the coordinate system which expects
arrays like ``(...,NDIM)`` to proceed with its calculations, the :py:class:`~pisces.geometry.handler.GeometryHandler` expects
only the "free axes" to be provided: ``(...,N_FREE)``.

Behind the scenes, :py:class:`~pisces.geometry.handler.GeometryHandler` coerces the coordinates you provide using the following
methods.

- :py:meth:`~pisces.geometry.handler.GeometryHandler.coerce_coordinates`: Converts partial coordinates to full coordinates.
- :py:meth:`~pisces.geometry.handler.GeometryHandler.coerce_coordinate_grid`: Reshapes coordinates into a structured grid.
- :py:meth:`~pisces.geometry.handler.GeometryHandler.coerce_function`: Wraps functions to operate only on free axes.


Performing Differential Operations
++++++++++++++++++++++++++++++++++

:py:class:`~pisces.geometry.handler.GeometryHandler` simplifies the computation of differential operations by managing
coordinate transformations and dependencies. Available operations include:

- :py:meth:`~pisces.geometry.handler.GeometryHandler.compute_gradient`: Computes the gradient of a scalar field.
- :py:meth:`~pisces.geometry.handler.GeometryHandler.compute_divergence`: Computes the divergence of a vector field.
- :py:meth:`~pisces.geometry.handler.GeometryHandler.compute_laplacian`: Computes the Laplacian of a scalar field.


Developer Notes / Subclass Implementation
-----------------------------------------

In Pisces, coordinate systems are always subclasses of the :py:class:`~pisces.geometry.base.CoordinateSystem` base-class. Each subclass implements
a couple of components to give it the correct behavior:

- :py:attr:`~pisces.geometry.base.CoordinateSystem.NDIM`: The number of dimensions that are characterized by the coordinate system.
- :py:attr:`~pisces.geometry.base.CoordinateSystem.AXES`: The names of the axes of the coordinate system.
- :py:attr:`~pisces.geometry.base.CoordinateSystem.PARAMETERS`: The parameters used to characterize the coordinate system.
- :py:meth:`~pisces.geometry.base.CoordinateSystem.to_cartesian` and :py:meth:`~pisces.geometry.base.CoordinateSystem.from_cartesian`: The transformations to and from
  cartesian coordinates.
- A Set of Lame coefficient functions.

With these basic building blocks, each coordinate system can independently determine its differential operators and
account for various other operations.

Class Initialization
''''''''''''''''''''

When you import the :py:mod:`~pisces.geometry` module, the first thing that happens is that the various :py:class:`~pisces.geometry.base.CoordinateSystem` subclasses
are generated. Each of these relies on the :py:class:`~pisces.geometry.base.CoordinateSystemMeta` class which does the following 3 critical things:

1. The axes of the :py:class:`~pisces.geometry.base.CoordinateSystem` subclass are converted to ``sympy`` symbols and stored in
   :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBAXES`.
2. The parameters in :py:attr:`~pisces.geometry.base.CoordinateSystem.PARAMETERS` are converted to ``sympy`` symbols and stored in the
   :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBPARAMS` dictionary.
3. The metaclass looks for the relevant **Lame coefficient functions** and passes the :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBAXES`
   and :py:attr:`~pisces.geometry.base.CoordinateSystem.SYMBPARAMS` symbols through them to generate symbolic expressions for each of the
   Lame coefficients.

   .. note::

        For developers, these symbolic Lame coefficients are stored in the ``_lame_symbolic`` attribute of each class.

The important idea in doing this is that the symbolic formulae for the Lame coefficients can be efficiently manipulated to
generate various important coefficients and parameters used in different differential operations. This ensures that we can
avoid having to do numerical operations where we could have derived a symbolic one instead, thus preserving accuracy at the
cost of a mild overhead.

Instantiation
'''''''''''''

Once a :py:class:`~pisces.geometry.base.CoordinateSystem` subclass exists, it stores the various symbolic expressions as functions of both the axes
(coordinate variables) and the parameters. When the user instantiates the class, they specify a specific value for each of the
parameters and thus "freeze" values into those symbols. This is all managed during the instantiation process for each class. Effectively,
when you create an instance of a :py:class:`~pisces.geometry.base.CoordinateSystem`, the following steps occur:

1. **Parameter Management**: The class ensures that the :py:attr:`~pisces.geometry.base.CoordinateSystem.PARAMETERS` are all provided by the user and / or
   are set to the class defaults. If the user provides an unexpected parameter, an error is raised.
2. **Lame Coefficient Setup**: The original Lame coefficient functions for the entire class are simplified using the user-provided values
   for the various parameters. Then they are converted to numerical versions using numpy operations.

   .. note::

        **Developer Note**: This creates two instance attributes: ``._lame_functions``, which holds the numerical forms
        of each of the Lame coefficients and ``._lame_inst_symbols`` which holds the symbolic versions once parameters are
        substituted in.

3. **Derived Attributes**: Depending on the class, there may be other symbolic / numeric attributes to derive. This is then carried
   out after the Lame coefficient functions are created.

   .. note::

        **Developer Note**: This process starts with the ``._derive_symbolic_attributes(self)`` method, which adds new sympy symbolic
        expressions to the ``self._symbolic_attributes`` dictionary. Once all of the symbolic attributes are generated, they are added
        to ``self._lambdified_attributes`` as callable functions. If a derived attribute is in the class-level attribute ``._SKIP_LAMBDIFICATION``,
        then they are not lambdified.


Subclassing
'''''''''''

Subclassing the :py:class:`~pisces.geometry.base.CoordinateSystem` requires careful implementation of several methods
and attributes to ensure the new coordinate system behaves correctly. Below are additional details for developers
implementing custom coordinate systems:

When creating a subclass, you must define specific methods to manage the transformation to and from Cartesian coordinates
and the Lame coefficient functions. These methods ensure compatibility with the broader Pisces library:

1. **Lame Coefficient Functions**:
   Every axis of the coordinate system must have a corresponding Lame coefficient function. These are implemented as methods
   in the subclass and are identified by the naming convention ``lame_<axis_index>``. For example, in a 3D coordinate system, the methods
   ``lame_0``, ``lame_1``, and ``lame_2`` would be defined for the first, second, and third axes, respectively. They should all have the signature

   .. code-block:: python

        class MyCoordinateSystem(CoordinateSystem):

            # ... All other implementation Details

            def lame_0(axes_0,axes_1, *args, parameter_0=None, parameter_1=None, **kwargs)
                return

   These methods must return a symbolic expression representing the Lame coefficient in terms of the symbolic axes and parameters.

2. **Coordinate Transformation Methods**:
   Each subclass must implement the following methods:

   - :py:meth:`_convert_native_to_cartesian`: Transforms native coordinates to Cartesian coordinates.
   - :py:meth:`_convert_cartesian_to_native`: Transforms Cartesian coordinates to the native coordinate system.

   These methods must handle all edge cases and should validate inputs to ensure they conform to expected shapes.
