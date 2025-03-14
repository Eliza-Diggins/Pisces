.. _modeling_overview:
Modeling Overview
===================

The cornerstone of the Pisces ecosystem is the generation of astrophysical models. These models, in turn, can be passed
on as initial conditions for simulations, used as test cases, or fed through instrument simulators to drive forward various
scientific endeavours. Unfortunately, physical modeling is not a simple undertaking and it is made more difficult by the
breadth of applicability that Pisces aims to achieve. In this reference guide, we will introduce you to Pisces models at the
level necessary to use the Pisces library for science.

.. tip::

    If you're a contributing developer, we suggest reading both this document and the :ref:`modeling_developer` document. We've
    also made an effort to exhaustively document the relevant codebase. The code is complex; however, it should be parsable
    given a solid understanding of python development.

Introduction
------------

Every model in Pisces is a subclass of the :py:class:`~pisces.models.base.Model` class; which handles most of the basic
logic of interaction, data management, file IO, etc. You're unlikely to ever need to interact directly with the model class; instead
you'll mostly need to interact with **subclasses** of :py:class:`~pisces.models.base.Model`, which define specific physical systems
like star clusters, galaxies, clusters of galaxies, etc.

Each of the :py:class:`~pisces.models.base.Model` subclasses provides the actual astrophysics that goes into the model and
provides a way to take a set of **inputs** provided by the user to a complete solution of the relevant physical properties of the
system. There are various components to the inputs, which are laid out in the next section.

The Model Domain
----------------

Pisces models operate within a defined spatial and physical domain, characterized by their coordinate systems, geometry,
boundaries, grids, scaling, and fields. Understanding these foundational aspects is crucial for effectively constructing
and utilizing models within the Pisces framework.

Coordinates, Geometry, and Boundaries
++++++++++++++++++++++++++++++++++++++

.. rubric:: Coordinates

A **coordinate system** defines the spatial framework over which a model is constructed. Pisces supports various coordinate
systems to accommodate different astrophysical scenarios (see :py:mod:`pisces.geometry` and the :ref:`geometry_overview` document).
It is up to the user to select the specific coordinate system (see :py:mod:`pisces.geometry.coordinate_systems`) that suits
the model they are trying to generate.

.. hint::

    Different model classes will permit different coordinate systems (see :py:attr:`~pisces.models.grids.base.ModelGridManager.ALLOWED_COORDINATE_SYSTEMS`).

    For example, a disk galaxy model probably only supports the :py:class:`~pisces.geometry.coordinate_systems.CylindricalCoordinateSystem` while
    elliptical galaxies probably support various flavors of the :py:class:`~pisces.geometry.coordinate_systems.PseudoSphericalCoordinateSystem`.

    It's not uncommon for a model to **only** support 1 geometry!

Whichever coordinate system you choose to use, it will determine which axes are available and tell the model how to perform
particular operations like integration, gradients, etc. necessary to solve the problem.

.. note::

    See the sections on fields (:ref:`grids_fields_scales`) and solving models (:ref:`solving`) for more details on
    how the coordinate system affects how the model is constructed.

.. rubric:: Boundaries

In addition to a coordinate system, A **bounding box** delineates the physical extent of the model within each coordinate axis.
It is defined as a pair of minimum and maximum values for each dimension, effectively creating an N-dimensional
"box" that contains the entire model domain.

.. note::

    The concept of a bounding box varies with the coordinate system. For instance, in spherical coordinates, the radial
    bounds define the inner and outer radii, while angular bounds specify the range of polar and azimuthal angles.

**Example**:

For a spherical coordinate system modeling a galaxy cluster:

.. code-block:: python

    bbox = [
        [1e-3, 100],      # Radial bounds: 0.001 to 100 kpc
        [0, np.pi],       # Polar angle bounds: 0 to π radians
        [0, 2 * np.pi]    # Azimuthal angle bounds: 0 to 2π radians
    ]


.. _grids_scales:
Grids, Scaling, and Fields
++++++++++++++++++++++++++++++++++++++

From a physical perspective, the **coordinate system** and the **bounding box** define the domain of the model; however,
Pisces models are generally not analytical objects. Instead, models store data (obtained while solving the relevant physics
problems) in "grids". The precise nature of the grid is another piece of information that the user can provide when setting
up a model.

.. tip::

    :py:class:`~pisces.models.base.Model` classes all have a linked :py:class:`~pisces.models.grids.base.ModelGridManager`
    object which handles the details of the grid, reading and writing data, etc.

    For a comprehensive explanation of the data infrastructure in models, see :ref:`model_grid_management`.

.. rubric:: Model Grids

Every model has a so-called **base grid** which *fills the domain* and *discretizes each axis* of the coordinate system.
Each model's grid manager has a :py:attr:`~pisces.models.grids.base.ModelGridManager.GRID_SHAPE`, which specifies how many **cells** are on each axis of
the coordinate domain.

**Example**

If you have an existing model (say ``example_model``), then

.. code-block:: python

    >>> print(example_model.GRID_SHAPE)
    [1000,10,10]

Would imply that there are 1000 points on the first axis, and 10 points on each of the other two axes.

.. hint::

    In a generic coordinate system with axes :math:`(x_1,\ldots,x_N)` and boundaries
    :math:`\left\{(x_{1,{\rm min}}, x_{1,{\rm max}}), \ldots, (x_{N,{\rm min}}, x_{N,{\rm max}})\right\}`, the grid will
    partition each coordinate range into some fixed number of **cells**.

.. rubric:: Scaling

Each axis of the coordinate system can also be given a scale, which determines how cells are spaced. Currently, only ``'linear'``
and ``'log'`` are valid spacings; however, these are each extremely useful in various contexts. For example, in a a galaxy
cluster model, its useful to have logarithmically spaced points on the :math:`r` axis while :math:`\theta` and :math:`\phi`
are linearly spaced.

.. _fields:
Fields
++++++

A **field** (:py:class:`~pisces.models.grids.base.ModelField`) is the atomic unit of all Pisces models. It represents a
physical quantity distributed across the model's grid. Thus, all the physical properties of a model are stored as fields on
the disk. The :py:attr:`~pisces.models.base.Model.FIELDS` attribute stores the fields in a model in a dictionary-like structure
so that they are easily accessed.

**Example**

In the case of a galaxy cluster model, you might have the following fields:

.. code-block:: python

    model = ClusterModel() # For example. Could be any model.

    # list the fields as the keys of model.FIELDS.
    for field_name,field in model.FIELDS.items():
        print(field_name)

    "gas_density"
    "dark_matter_density"
    "gravitational_potential"
    "temperature"
    "pressure"
    "stellar_density"
    "entropy"
    "..."

.. raw:: html

   <hr style="height:2px;background-color:black">

Field Domains
'''''''''''''

Fundamentally, each field is an **array of data on disk**. The array corresponds to a "slice" of the base grid, which selects
only the relevant axes. Thus, in a 3-D spherical model, a radially symmetric field would only be stored as an array over the
1-D set of radii in the grid. A rotationally symmetric field (about the :math:`z`-axis) would have a 2-D field over the
:math:`r` and :math:`\theta` axes of the grid.

.. note::

    While fields may occupy specific axes of a grid, whichever axes they do occupy are **fully occupied**, meaning that
    the axes uniquely define the shape of the underlying data grid.

Fields are managed as disk-backed arrays, allowing for efficient
memory usage by loading only necessary slices into memory.

Reading Fields from Disk
''''''''''''''''''''''''

On the disk, every :py:class:`~pisces.models.grids.base.ModelField` is an ``HDF5`` dataset in the model's file. When you
load a :py:class:`~pisces.models.base.Model` instance, the class will look through the file structure and generate pointers
to the fields stored there. Fields are not fully loaded into memory upon model initialization. Instead, Pisces utilizes
a **lazy loading** strategy, where data is fetched from disk only when explicitly accessed. This approach minimizes memory
consumption, particularly beneficial when dealing with large datasets.

Each field in Pisces has a dual identity:

1. **On Disk**: Represented as an HDF5 dataset with a specific shape corresponding to the grid's dimensions.
2. **In Memory**: Initially a lightweight pointer (zero-size) to the HDF5 dataset. Only when a slice of the field is
   accessed does Pisces read the relevant data into memory.

.. important::

    When you slice into a :py:class:`~pisces.models.grids.base.ModelField`, the output is an ``unyt.unyt_array`, which
    handles the units of the field seamlessly.

**Example: Accessing and Manipulating Fields**

.. code-block:: python

    # Accessing fields from the model
    density_field = model.FIELDS["gas_density"]
    temperature_field = model.FIELDS["temperature"]

    # Loading entire arrays into memory
    density = density_field[...]
    temperature = temperature_field[...]

    # Computing ideal gas pressure (excluding constants)
    pressure = density * temperature

    # Adding the pressure field to the model
    model.add_field_from_function(lambda x, y, z: pressure, "pressure")

**Example: Slicing Fields**

When a field is sliced, Pisces determines the necessary portion of the data to load based on the requested indices.

.. code-block:: python

    # Load a specific slice of the density field into memory
    density_slice = density_field[50:60, :, :]

    # Perform computations on the slice
    average_density = np.mean(density_slice)

    # Update the model with the modified slice
    density_field[50:60, :, :] = average_density

**Advantages of Lazy Loading**

- **Memory Efficiency**: Only the required data segments are loaded, preventing memory overload.
- **Performance Optimization**: Reduces initial load times by deferring data access until necessary.
- **Scalability**: Facilitates handling of extremely large models that exceed available RAM.

Memory Safety and Chunking
''''''''''''''''''''''''''

Memory management is a critical aspect of handling large-scale astrophysical models. Pisces employs a strategic
approach to memory safety through **chunking**, enabling efficient processing of vast datasets without overwhelming system
resources.

**Chunking** refers to partitioning the entire grid domain into smaller, more manageable sub-arrays or *chunks*.
Each chunk is an :math:`N_\mathrm{dim}`-dimensional sub-region of the domain, containing a subset of the cells along every axis.

**Purpose of Chunking**

1. **Memory Efficiency**:
   Large grids (e.g., thousands of cells per axis) can consume significant memory. By dividing the grid into chunks,
   Pisces ensures that only a fraction of the data is loaded into memory at any given time, preventing memory overflows.

2. **Parallel Workflows**:
   Chunk-based iteration facilitates distributed computing. Each chunk can be processed independently, allowing for
   parallel execution across multiple processors or nodes.

3. **I/O Performance**:
   Modern HDF5 libraries optimize chunked datasets for partial reads and writes. Accessing a specific chunk reduces
   disk I/O overhead by loading only the necessary portion of the dataset.

Pisces allows users to define the chunk shape during grid manager initialization. The chunk shape determines the size of each chunk along every axis.

**Example: Initializing a Grid Manager with Chunking**

.. code-block:: python

    from pisces.models.grids.base import ModelGridManager
    from pisces.geometry.coordinate_systems import CartesianCoordinateSystem

    # Define grid parameters
    coordinate_system = CartesianCoordinateSystem()
    bbox = [[-100, 100], [-100, 100], [-100, 100]]  # x, y, z boundaries
    grid_shape = [1000, 1000, 1000]                # High-resolution grid
    chunk_shape = [100, 100, 100]                  # Define chunk size

    # Initialize the grid manager with chunking
    manager = ModelGridManager(
        path="large_grid.h5",
        coordinate_system=coordinate_system,
        bbox=bbox,
        grid_shape=grid_shape,
        chunk_shape=chunk_shape,
        scale=['linear', 'linear', 'linear'],
        overwrite=True
    )

**Constraints on Chunk Shape**

- **Divisibility**:
  Each element of the `chunk_shape` must evenly divide the corresponding element in `grid_shape`. This ensures
  uniform chunk sizes and simplifies I/O operations. Pisces does not support partial or irregular chunking.
  All chunks must be the same size across the grid.

**Example: Valid and Invalid Chunk Shapes**

.. code-block:: python

    # Valid chunk shape (evenly divides grid_shape)
    grid_shape = [1000, 1000, 1000]
    chunk_shape = [100, 100, 100]  # Valid

    # Invalid chunk shape (does not evenly divide grid_shape)
    chunk_shape = [333, 100, 100]  # Invalid, as 333 does not divide 1000 evenly


Pisces provides methods to iterate over and manipulate chunks efficiently. These methods abstract the underlying
complexity, allowing users to focus on computations rather than data partitioning.

**Iterating Over Chunks**

.. code-block:: python

    # Iterate over all chunks in the grid
    for chunk_index in manager.iterate_over_chunks():
        chunk_mask = manager.get_chunk_mask(chunk_index)
        # Access the density field slice corresponding to the current chunk
        density_chunk = manager.FIELDS["gas_density"][tuple(chunk_mask)]

        # Perform computations on the chunk
        processed_density = process_density(density_chunk)

        # Update the field with the processed data
        manager.FIELDS["gas_density"][tuple(chunk_mask)] = processed_density

**Processing Chunks in Parallel**

By processing chunks independently, Pisces models can leverage parallel computing resources to accelerate computations.

.. code-block:: python

    import multiprocessing as mp

    def process_chunk(chunk_index, manager):
        chunk_mask = manager.get_chunk_mask(chunk_index)
        density_chunk = manager.FIELDS["gas_density"][tuple(chunk_mask)]
        # Perform some intensive computation
        processed_density = intensive_computation(density_chunk)
        manager.FIELDS["gas_density"][tuple(chunk_mask)] = processed_density

    # Initialize multiprocessing pool
    pool = mp.Pool(processes=4)  # Number of parallel workers

    # Distribute chunk processing across workers
    pool.starmap(process_chunk, [(chunk_idx, manager) for chunk_idx in manager.iterate_over_chunks()])

    pool.close()
    pool.join()

**Benefits of Chunk-Based Processing**

- **Scalability**:

  Distributes computational load, allowing models to scale with available hardware resources.

- **Efficiency**:

  Reduces memory footprint by ensuring only a manageable portion of data is in memory at any given time.

- **Flexibility**:

  Enables users to tailor processing strategies based on specific computational needs and constraints.



**Example: Safe Field Access**

.. code-block:: python

    # Access a large field safely by loading only necessary chunks
    density_field = model.FIELDS["gas_density"]

    # Iterate over chunks to compute the mean density without loading the entire field
    total_density = 0.0
    count = 0

    for chunk_index in model.grid_manager.iterate_over_chunks():
        chunk_mask = model.grid_manager.get_chunk_mask(chunk_index)
        density_chunk = density_field[tuple(chunk_mask)]
        total_density += np.sum(density_chunk)
        count += density_chunk.size

    mean_density = total_density / count
    print(f"Mean Gas Density: {mean_density}")

**Error Prevention**

By enforcing strict chunking rules and providing comprehensive error messages, Pisces helps users avoid common pitfalls related to memory management.

- **Invalid Chunk Shapes**:

  Attempting to define a chunk shape that does not evenly divide the grid shape results in clear and informative errors, preventing inconsistent data partitioning.

- **Out-of-Bounds Access**:

  Accessing data outside the defined grid boundaries is detected and reported, safeguarding against inadvertent memory access violations.

**Example: Handling Chunk Shape Errors**

.. code-block:: python

    from pisces.models.grids.base import ModelGridManager
    from pisces.geometry.coordinate_systems import CartesianCoordinateSystem

    # Define grid parameters
    coordinate_system = CartesianCoordinateSystem()
    bbox = [[-50, 50], [-50, 50], [-50, 50]]
    grid_shape = [1000, 1000, 1000]
    chunk_shape = [333, 100, 100]  # Invalid chunk shape

    try:
        manager = ModelGridManager(
            path="invalid_chunk_grid.h5",
            coordinate_system=coordinate_system,
            bbox=bbox,
            grid_shape=grid_shape,
            chunk_shape=chunk_shape,
            scale=['linear', 'linear', 'linear'],
            overwrite=True
        )
    except ValueError as e:
        print(f"Failed to initialize grid manager: {e}")

**Output**:

.. code-block:: text

    Failed to initialize grid manager: The chunk shape [333, 100, 100] does not evenly divide the grid shape [1000, 1000, 1000].
    Each chunk dimension must be a divisor of the corresponding grid dimension.


To maximize the benefits of chunking while maintaining memory safety and computational efficiency, consider the following best practices:

1. **Choose Optimal Chunk Sizes**:

   - Balance between memory usage and I/O overhead. Smaller chunks reduce memory consumption but may increase disk access frequency.
   - Align chunk sizes with the grid's structure and the nature of computational operations.

2. **Leverage Parallel Processing**:

   - Utilize multiprocessing or distributed computing frameworks to process multiple chunks concurrently, accelerating computations.

3. **Monitor Memory Usage**:

   - Regularly assess memory consumption, especially when dealing with exceptionally large grids or complex computations.

4. **Avoid Excessive Slicing**:

   - Minimize the number of slice operations to reduce the overhead associated with frequent disk reads and writes.

5. **Utilize Bulk Operations When Possible**:

   - For operations that span multiple chunks or the entire grid, consider loading larger data segments into memory to optimize performance.

6. **Implement Robust Error Handling**:

   - Anticipate and handle potential errors related to data access, chunking constraints, and memory limitations to ensure model integrity.

.. raw:: html

   <hr style="height:2px;background-color:black">

Model Attributes and Components
-------------------------------

Having become familiar which the general structure of the model domain, the base grid, and the various components of the
backend, its time to look at the interface that the model provides to the user. There are 3 core components of the model
for the user to interact with and utilize for their scientific needs:

1. The **Grid Manager**
2. The **Field Container**
3. The **Profiles**

Each of the following sections covers one of these components.

The Grid Manager
++++++++++++++++++

.. tip::

    Please look at the :ref:`model_grid_management` more a more comprehensive look at :py:class:`~pisces.models.grids.base.ModelGridManager`
    classes.

The **Grid Manager** is a pivotal component of the Pisces modeling infrastructure. It encapsulates all aspects related to the
spatial grid, including coordinate systems, grid configuration, and data storage mechanisms. By abstracting these details, the
Grid Manager allows users to focus on the physical modeling without delving into the complexities of data management.

.. rubric:: Responsibilities

- **Coordinate System Access**:

  The Grid Manager provides access to the model's coordinate system, ensuring that all spatial operations are consistent with
  the chosen framework. This includes handling transformations, integrations, gradients, and other differential operations
  essential for solving physical equations.

  .. tip::

      You can use the combination of the **coordinate system** and a fields axes (:py:attr:`~pisces.models.grids.base.ModelField.AXES`)
      to construct a :py:class:`~pisces.geometry.handler.GeometryHandler` to perform various calculations in the specific
      geometry of your system.

- **Grid Configuration**:

  Manages the grid's spatial discretization, including the number of cells along each axis, cell sizes, and scaling factors
  (linear or logarithmic). This configuration directly impacts the resolution and accuracy of the model.

- **Chunk Management**:

  Implements chunking strategies to partition the grid into manageable sub-regions, facilitating efficient memory usage and
  parallel processing. By controlling chunk sizes, the Grid Manager optimizes data access patterns for performance.

- **Data Storage and Access**:

  Oversees the storage of physical fields on disk using the HDF5 format. It ensures that data is organized, accessible, and
  efficiently retrievable, leveraging HDF5's capabilities for handling large datasets.

.. rubric:: Accessing the Grid Manager

Users typically interact with the Grid Manager indirectly through the model's interface. However, understanding how to access
and utilize the Grid Manager can enhance model customization and performance tuning.

**Example: Accessing Grid Manager Attributes**

.. code-block:: python

    # Instantiate a model
    from pisces.models.galaxy_clusters import ClusterModel
    model = ClusterModel("path/to/cluster_model.h5")

    # Access grid manager
    grid_manager = model.grid_manager

    # Retrieve coordinate system
    coord_system = grid_manager.coordinate_system
    print(coord_system)

    # Retrieve grid shape
    grid_shape = grid_manager.GRID_SHAPE
    print(f"Grid Shape: {grid_shape}")

    # Retrieve chunk shape
    chunk_shape = grid_manager.CHUNK_SHAPE
    print(f"Chunk Shape: {chunk_shape}")

.. note::

    Implementing custom grid managers requires a deep understanding of Pisces' grid management system. Ensure that any
    modifications maintain consistency with the coordinate system and adhere to Pisces' data management protocols to prevent
    data corruption or access issues.

.. note::

    The **field container**, which we cover in the next section is actually part of the :py:class:`~pisces.models.grids.base.ModelGridManager`.
    Behind the scenes, the :py:attr:`pisces.models.base.Model.FIELDS` property is just a reference to the model's
    :py:class:`~pisces.models.grids.base.ModelGridManager.FIELDS` element.

The Field Container
+++++++++++++++++++

The **Field Container** is an integral part of the model, providing a streamlined interface for accessing, adding,
removing, and managing physical fields within the model. It abstracts the underlying data storage, allowing users to interact
with fields as if they were standard Python dictionaries while leveraging the efficiency and scalability of HDF5-backed
datasets.

Fields are accessed through the :py:attr:`~pisces.models.base.Model.FIELDS` attribute of the model, which references
the :py:class:`~pisces.models.grids.base.ModelFieldContainer` within the Grid Manager. This container behaves like a
dictionary, mapping field names to their corresponding data arrays.

**Example: Listing Available Fields**

.. code-block:: python

    # Instantiate a model
    from pisces.models.base import ClusterModel
    model = ClusterModel("path/to/cluster_model.h5")

    # List all available fields
    for field_name, field in model.FIELDS.items():
        print(field_name)

    # Output:
    # gas_density
    # dark_matter_density
    # gravitational_potential
    # temperature
    # pressure
    # stellar_density
    # entropy
    # ...

Adding and Removing Fields
''''''''''''''''''''''''''

Users can add new fields to the model by defining functions, utilizing profiles or directly setting arrays into the dataset.
The Field Container provides methods to facilitate these operations, ensuring that new fields are correctly integrated into the grid's structure.

**Adding a Field from a Function**: [See :py:meth:`~pisces.models.base.Model.add_field_from_function`]

.. code-block:: python

    # Define a function to compute pressure
    def compute_pressure(x, y, z):
        return model.FIELDS["gas_density"][x, y, z] * model.FIELDS["temperature"][x, y, z]

    # Add the pressure field to the model
    model.add_field_from_function(compute_pressure, "pressure")

**Adding a Field from a Profile**: [See :py:meth:`~pisces.models.base.Model.add_field_from_profile`]

.. code-block:: python

    # Assume 'temperature_profile' is a registered profile
    model.add_field_from_profile("temperature_profile", "temperature")

**Adding A Generic Field** [See :py:meth:`~pisces.models.grids.base.ModelFieldContainer.add_field`]

.. code-block:: python

    # Add a 1D radial temperature dataset.
    model.FIELDS.add_field("temperature", ["r"], data=data_array)

Fields can be removed from the model using the `del` statement or the :py:meth:`~pisces.models.base.ModelFieldContainer.remove_field` method.
This is useful for cleaning up unnecessary data or replacing fields with updated versions.

**Example: Removing a Field**

.. code-block:: python

    # Remove the 'entropy' field from the model
    del model.FIELDS["entropy"]

    # Alternatively, using the remove_field method
    model.FIELDS.remove_field("entropy")

.. note::

    Removing a field permanently deletes its data from the HDF5 file. Ensure that you have backups or are certain about
    the removal before proceeding.

Listing and Inspecting Fields
'''''''''''''''''''''''''''''

:py:class:`~pisces.models.grids.base.ModelField` instances have a variety of attributes which are useful when working with them.
Most importantly, the units can be accessed using the :py:attr:`~pisces.models.grids.base.ModelField.units` attribute and
the shape can be accessed using the typical ``.shape`` attribute.

**Example: Inspecting Field Properties**

.. code-block:: python

    # Retrieve properties of a specific field
    pressure_field = model.FIELDS["pressure"]
    print(f"Units: {pressure_field.units}")
    print(f"Data Type: {pressure_field.dtype}")
    print(f"Shape: {pressure_field.shape}")

    # Output:
    # Units: erg/cm³
    # Data Type: float64
    # Shape: (1000, 10, 10)


Profiles
++++++++

Profiles in Pisces are fundamental components that define the analytical distributions of physical quantities within a model. Unlike fields,
which store discretized data on the grid, profiles represent continuous mathematical functions such as density,
temperature, or velocity distributions. This separation ensures that profiles remain lightweight and flexible,
facilitating efficient initialization and manipulation of model fields.

.. tip::

    Please look at the :ref:`profiles-overview` more a more comprehensive look at the :py:mod:`~pisces.profiles`
    module.

A **Profile** encapsulates an analytical function that describes a physical quantity's distribution across the model's domain.
Profiles are defined as subclasses of the abstract base class :py:class:`~pisces.profiles.base.Profile`, which provides
the necessary framework for symbolic and numerical operations. Each profile is characterized by its independent variables (axes),
parameters, and the functional form that defines its behavior.

Profiles and Models
'''''''''''''''''''

:py:class:`~pisces.profiles.base.Profile` class instances are stand-alone objects and useful in their own right; however,
they are frequently used as tools when creating models.

.. hint::

    In many cases, they are the starting inputs for certain models.

Because they play an important role in constructing models, it is useful to retain their high precision and easy usability instead
of simply interpolating them onto a field. As such, every :py:class:`~pisces.models.base.Model` instance has an attached registry of
profiles (a :py:class:`~pisces.profiles.collections.HDF5ProfileRegistry`) which keeps track of any relevant profiles the
user (or developer) chooses to store in the model. You can access the profiles attached to a model using the :py:attr:`~pisces.models.base.Model.profiles`
attribute.

.. tip::

    These are stored separately from the model fields and using a different format (within the same HDF5 file). Consult the documentation on
    the :py:class:`~pisces.profiles.base.Profile` class for details.

.. tip::

    In the deep recesses of the Pisces code base, there is a really useful class; the :py:class:`pisces.io.hdf5.HDF5ElementCache`. This
    is the base class for both :py:class:`~pisces.profiles.collections.HDF5ProfileRegistry` and :py:class:`~pisces.models.base.ModelFieldContainer`.
    The base class facilitates creating these sorts of "repositories" in HDF5 and supports lazy loading for memory efficiency.

Adding and Removing Profiles
''''''''''''''''''''''''''''

Any instance of a :py:class:`~pisces.profiles.base.Profile` can be registered to your model's profile's repository using
``model.profiles.add_profile`` (:py:meth:`~pisces.profiles.collections.HDF5ProfileRegistry.add_profile`) and can be removed
using the standard ``del ...`` notation.

The HDF5 File Structure
+++++++++++++++++++++++

Below is a schematic representation of the Pisces model's HDF5 data structure. This hierarchical organization ensures
efficient storage, access, and management of the model's spatial and physical data.

.. code-block:: ascii

    /MODEL
    ├── /FIELDS
    │   ├── gas_density (dataset)
    │   ├── dark_matter_density (dataset)
    │   ├── gravitational_potential (dataset)
    │   ├── temperature (dataset)
    │   ├── pressure (dataset)
    │   ├── stellar_density (dataset)
    │   └── entropy (dataset)
    ├── /PROFILES
    │   ├── radial_density (group)
    │   ├── temperature_profile (group)
    │   └── pressure_profile (group)
    ├── /CSYS
    │   └── coordinate_system (group)
    └── /ATTRIBUTES
        ├── bounding_box (attribute)
        ├── grid_shape (attribute)
        ├── chunk_shape (attribute)
        └── scale (attribute)

.. note::

    This schematic provides an overview of the hierarchical organization within the HDF5 file used by Pisces models.
    Each group and dataset plays a specific role in managing the model's spatial and physical data, ensuring scalability
    and efficiency for large-scale astrophysical simulations.

.. _solving:
Model Physics and Solving
-------------------------

In Pisces, the process of defining a physical model involves not only specifying the spatial and physical distributions
through profiles but also orchestrating the computational steps required to solve the model.
This section delves into the mechanisms behind model solving, emphasizing **Solution Pathways** and the **Solving Process**.

Solution Pathways
++++++++++++++++++

A **Solution Pathway** is a predefined sequence of computational steps that transform a model from an initial state to
a fully solved state. Each pathway encapsulates a series of processes and validation checks that ensure the model evolves
correctly and consistently. By structuring the solving process into pathways, Pisces promotes modularity, reusability,
and clarity in model computations.

.. note::

    The details of writing models is covered in :ref:`modeling_developer`. For ordinary users of the code, the
    generation pathways are already built in.

Each **solution pathway** in a class has a name (see :py:meth:`~pisces.models.base.Model.list_pathways`) by which you
can refer to it. To solve a model using a specific pathway, you can specify the pathway name in the ``__call__`` to
the model class (or using :py:meth:`~pisces.models.base.Model.solve_model`).

The specified pathway has two core components:

1. The **checker(s)** (or validator(s)): check that your :py:class:`~pisces.models.base.Model` instance meets the necessary
   conditions for the specified pathway to work.
2. The **process(es)**: are a series of functions / methods which perform physical operations on the model and create
   fields, profiles, etc. that are necessary.

For example, a galaxy cluster pathway might check the initial model for the temperature and density profile and then (if it
passes validation) uses those to compute the pressure. With the pressure it might then compute the gravitational field, etc.
Thus, the steps / **processes** define the physical pipeline through which the model ends up being solved.

.. tip::

    Models may have many pathways which intersect, overlap, or call the same process multiple times. This allows a single
    model to manage things like different equations of state, different gravities, and different geometries.

.. note::

    In general, a single :py:class:`~pisces.models.base.Model` is defined for as many different permutations on a single physical
    system as possible. If a specific case of the model requires a sufficient amount of additional work to solve, it may be created
    as an entirely new model subclass.

    Generally, the same physical system but with different coordinate systems should be the same model, while different
    assumptions about underlying physics (say a magnetized or rotating galaxy cluster ICM) might be seperate classes.

Solving a Model
+++++++++++++++

To solve a model, invoke the :py:meth:`~pisces.models.base.Model.solve_model` method, specifying the desired pathway.
The solver will then execute each process in the pathway, adhering to the defined step order and passing any specified arguments.

.. note::

    Under the hood, the :py:class:`~pisces.models.base.Model` has a :py:class:`~pisces.models.solver.ModelSolver` instance
    attached to it which manages the validation step and execution of the pathway.

Before executing a pathway, the solver performs validation checks to ensure that all necessary conditions are satisfied.
This prevents the solver from running processes that might lead to inconsistent or erroneous model states.

.. rubric:: Generator Processes

For most models, the user never needs to actually invoke the solver directly, instead they may use a so-called **generator method**.
The **generator methods** are ``classmethods`` which take a specific set of arguments necessary to use a particular pathway and
then (in one pass) create the skeleton for the model and run the solver. This mean that, for most cases, the action of creating
the model skeleton and then solving it are done at the same time.
