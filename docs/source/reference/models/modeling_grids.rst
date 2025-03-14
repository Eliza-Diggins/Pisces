.. _model_grid_management:
Model Data Structures
=====================

Underlying every physical model in pisces (from the :py:mod:`~pisces.models` module), is a so-called "grid manager" which
ensures that the coordinate system, physical data, units, and solution pipelines at the model level are successfully coordinated
to write data to a standardized HDF5 format. This is a complex aspect of the Pisces ecosystem and is generally abstracted
to a degree that most users do not need to be overly familiar with it. Nonetheless, this document will summarize the details
of how grid managers work and the scope of their activity.

Model Fields
------------

The atomic unit of :py:class:`~pisces.models.base.Model` classes is the "field" (:py:class:`~pisces.models.grid.base.ModelField`),
which represents a particular physical property of the model over the model's domain.

.. admonition:: What is a Field?
   :class: tip

   In Pisces, a **Field** (see :py:class:`~pisces.models.grids.base.ModelField`) is effectively an array of data
   describing some physical quantity (e.g., density, temperature) on a portion
   (or the entirety) of the grid. Each field is a subclass of ``unyt.unyt_array``
   but references an HDF5 dataset on disk — allowing slices/chunks to be loaded
   on-demand without forcing the entire dataset into memory.

Accessing Fields
++++++++++++++++

When a :py:class:`~pisces.models.grids.base.ModelField` is created or retrieved it has a dual-identity. On the HDF5 file it abstracts, it becomes
a fully-fledged HDF5 dataset with the specified grid size. In memory, it starts as a 0-size pointer to the HDF5 dataset object.
Slicing into the field (e.g., ``field[10:20]``) triggers an actual disk read. This architecture preserves memory by avoiding full-array
loads, especially important for large 3D or 4D datasets.

.. note::

   Internally, fields are managed by a :py:class:`~pisces.models.grids.base.ModelFieldContainer` attached
   to the parent :py:class:`~pisces.models.grids.base.ModelGridManager`. You can create or retrieve fields
   through dictionary-like operations on the container (e.g., ``manager.FIELDS["my_field"]``).

   See the section below on managers.

.. tip::

    The flip-side of this memory-conserving architecture is that, if slicing is done naively, it can be inefficient
    in terms of execution time. In many cases, it is fine (and encouraged) to just read the entire field at once when
    loading it. The memory conservation architecture is only relevant for 3D or higher dimensional grids where sufficient
    resolution requires memory loads larger than most personal computers.

Working With Fields
+++++++++++++++++++

When you have a particular model field, it should be have very similarly to a standard :py:class:`unyt.unyt_array` object.
If you cut into it, you'll get an :py:class:`unyt.unyt_array` back, so you can perform operations like the following:

.. dropdown:: Example

    .. code-block:: python

        # pressure and temperature fields obtained from a profile or from a model...
        density_field, temperature_field = ModelField(...), ModelField(...)

        # load the unyt_arrays from the underlying field.
        df, tf = density_field[...], temperature_field[...]

        # compute the ideal gas pressure (excluding a factor of m_p * mu)
        pressure = df*tf

        # Proceed with additional computations or add the pressure to an HDF5 file.

Naturally, models have the :py:attr:`~pisces.models.base.Model.FIELDS` attribute, which actually connects you to a dictionary like
collection of fields which can then be accessed using string keys.

All fields have units (like their ``unyt.unyt_array`` counterparts), and come in specific ``dtypes``.

.. raw:: html

   <hr style="height:2px;background-color:black">

Grid Managers
-------------

Between the :py:class:`~pisces.models.base.Model` instance and the individual fields is is the "grid manager"
(:py:class:`~pisces.models.grids.base.ModelGridManager`). The grid manager is in charge of the details of data storage,
field creation and deletion, the physical domain of the model, etc.

.. important::

    The most important idea to keep track of here is that there are 3 layers of abstraction at play:

    1. :py:class:`~pisces.models.base.Model` is the most abstract layer.
       This layer cares about actually solving the physics problems necessary to generate the model.
       The details of the coordinate system, grid management, etc. are all delegated further down the hierarchy.
    2. :py:class:`~pisces.models.grids.base.ModelGridManager` is the middle layer.
       The job of this layer is to deal with the coordinate system, the base grid, the boundary box, chunking, and
       all of the other details of data access and storage.
    3. :py:class:`~pisces.models.grids.base.ModelField` is the least abstract layer.
       The job of this layers is to simply act as a dynamically loaded container for the underlying physical data
       of the model.

A :py:class:`~pisces.models.grids.base.ModelGridManager` ensures the following:

- A consistent bounding box for each axis (:py:attr:`~ModelGridManager.BBOX`).
- A uniform shape and chunk shape for partial I/O or chunked operations.
- Unified HDF5-based backend, storing all array data in one or more datasets.

Thanks to the manager, any code that needs to read or write part of a field can do so
with minimal overhead, and without manually tracking slices or bounding boxes.

Components of the Manager
+++++++++++++++++++++++++

There are a number of components in the :py:class:`~pisces.models.grids.base.ModelGridManager`:

1. **Coordinate System** (:py:class:`~pisces.geometry.base.CoordinateSystem`):
   The coordinate system determines the number of dimensions in the base grid (:py:attr:`~pisces.geometry.base.CoordinateSystem.NDIM`),
   the available axes (:py:attr:`~pisces.geometry.base.CoordinateSystem.AXES`), and other details of the underlying geometry.

   Higher up the hierarchy, the coordinate system determines how the model solution pipelines compute things like
   the gradient, or divergence.

2. **Metadata**:
   The :py:class:`~pisces.models.grids.base.ModelGridManager` also carries a considerable amount of metadata about the model:

   - The **bounding box** for the model (:py:attr:`~pisces.models.grids.base.ModelGridManager.BBOX`)
     The bounding box is the "box" in coordinate space which contains the entire physical domain of the model.

     .. tip::

        The bounding box isn't actually a box unless you're working in cartesian coordinates!

   - The **grid** and **chunk shapes** (:py:attr:`~pisces.models.grids.base.ModelGridManager.GRID_SHAPE` and :py:class:`~pisces.models.grids.base.ModelGridManager.CHUNK_SHAPE`)
     The grid and chunk sizes determine how "fine" the resolution of the underlying grid is. The grid shape in particular specifies the
     total number of cells in the domain. The chunk shape is only relevant for chunked operations, but it controls the size of individual
     computational chunks in the grid space.

     .. note::

        Large :py:attr:`~pisces.models.grids.base.ModelGridManager.GRID_SHAPE`-s will correspond with slower computation times
        but better detail / resolution. Depending on the model and the mathematics involved, this could have an impact on the
        reliability of results.

   - Other, more minor, metadata:

     .. seealso::

        :py:attr:`~pisces.models.grids.base.ModelGridManager.CELL_SIZE`
        :py:attr:`~pisces.models.grids.base.ModelGridManager.scale`
        :py:attr:`~pisces.models.grids.base.ModelGridManager.SCALED_BBOX`


3. **Fields**: The :py:class:`~pisces.models.grids.base.ModelGridManager` has the :py:attr:`~pisces.models.grids.base.ModelGridManager.FIELDS` attribute,
   which is a container of :py:class:`~pisces.models.grids.base.ModelField` instances which behaves like a dictionary. Thus, you can access a specific
   field from the manager as

   .. code-block:: python

        grid_manager = ... # Some grid manager from a model or other construction method.
        density_array = grid_manager.FIELDS['gas_density'][...]

        # notice the ... indexes into the ModelField to get unyt.unyt_array.


Creating a Grid Manager
+++++++++++++++++++++++

When building a grid manager from scratch, you must ensure that you have two critical pieces of information:

- A valid **coordinate system**, and
- Enough information to define the physical domain.

Generally speaking, the physical domain is composed of a couple components:

- **bbox**: A bounding box in the form of a ``(2, NDIM)`` array (or a Python list of lists), specifying the minimum and maximum
  coordinates along each axis.
- **grid_shape**: A tuple or list of integers of length :math:`N_\mathrm{dim}`, giving the number of cells along each axis.
- *optional* **chunk_shape**: Allows chunk-based memory usage (see below), but if not provided, defaults to `grid_shape`.
- *optional* **scale**: The scaling of each axis. If an axis has ``"log"`` scale, then the grid is evenly spaced in log scale. Otherwise
  it is evenly spaced in linear scale.

For example, we can create a spherical grid manager with a logarithmic radial coordinate as

.. code-block:: python

    import numpy as np
    from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
    from pisces.models.grids.base import ModelGridManager

    # Create the coordinate system, the bounding box, the grid shape, etc.
    coord_system = SphericalCoordinateSystem()
    bbox = [[1e-3,100],[0,np.pi],[0,2*np.pi]] # r: (0,1), theta: (0, np.pi), phi: (0,2pi)
    grid_shape = [100,10,10]
    scaling = ['log','linear','linear']

    # Create the manager at the test.hdf5 file location
       manager = ModelGridManager(
       "my_grids.h5",
       coordinate_system=coord_system,
       bbox=bbox,
       grid_shape=grid_shape,
       overwrite=True,
       scale=scaling
   )

.. tip::

    When these parameters are given, the manager calls its "skeleton builder" (:py:meth:`~pisces.models.grids.base.ModelGridManager.build_skeleton`),
    which creates a new HDF5 file (or overwrites an existing one, if ``overwrite=True`` was set), and populates
    the file with metadata about the bounding box, chunking, coordinate system, and more.


The manager automatically stores:

- A **coordinate system** object, so that the code or user can retrieve the model’s axes or geometry specifics later.
- The **domain extent** (bounding box) as an attribute in HDF5, indicating the physical region.
- **Grid shape** as a 1D array in HDF5, tying each dimension to its axis from the coordinate system.
- (Optionally) a **chunk shape** specifying how the domain is subdivided in memory.

.. tip::

   The bounding box does not need to be "box-shaped" in a geometric sense if you are using a specialized coordinate system
   (like spherical). The bounding box entries simply define the minimum and maximum allowed values of each axis in that system.
   For instance, you might have bounding box entries for ``(r_min, r_max)``, ``(theta_min, theta_max)``, ``(phi_min, phi_max)``
   if you have a spherical coordinate system.

By default, if the file at ``path`` already exists and you do **not** specify ``overwrite=True``, the manager attempts
to open and load that file’s existing skeleton. In that case, the coordinate system and bounding box are inferred from
the HDF5 metadata.

**Advanced Usage**:

- **length_unit** and **scale** can also be provided to specify the physical unit (e.g., ``'kpc'``, ``'m'``) and whether
  each axis is ``'linear'`` or ``'log'``. If omitted, the defaults on the class
  (:py:attr:`~pisces.models.grids.base.ModelGridManager.DEFAULT_LENGTH_UNIT` and :py:attr:`~pisces.models.grids.base.ModelGridManager.DEFAULT_SCALE`) are used.

- If you want to do more sophisticated initialization (like applying constraints on the bounding box or hooking into
  model-specific metadata), you can subclass :py:class:`~pisces.models.grids.base.ModelGridManager` and override :py:meth:`_load_attributes`
  or :py:meth:`_compute_secondary_attributes`. This pattern is used by some specialized simulation codes.

Chunking
++++++++

.. note::

    **Chunking** refers to partitioning the entire grid domain into smaller, more manageable sub-arrays or *chunks*.
    Each chunk is an :math:`N_\mathrm{dim}`-dimensional sub-region of the domain, containing a subset of the cells
    along every axis.

Here’s why chunking is relevant:

1. **Memory Efficiency**: For large grids (e.g., hundreds or thousands of cells in each dimension),
   loading or operating on the entire field array can exceed available RAM. By dividing the domain into
   chunks, you can process only one chunk at a time, using a fraction of the memory.

2. **Parallel Workflows**: Some advanced models or scripts might process each chunk in a separate worker or
   node. Chunk-based iteration allows you to seamlessly distribute workload.

3. **I/O Performance**: Modern HDF5 libraries can handle chunked datasets efficiently when partial reads
   and writes are needed. If you only need a slice from the array, chunking can reduce the overhead
   by reading just the relevant portion on disk.

.. code-block:: python

   from pisces.models.grids.base import ModelGridManager, ChunkIndex

   manager = ModelGridManager(
       path="my_grids.h5",
       # ...
       chunk_shape=[50, 100]
   )

   # Suppose the total grid shape is [100, 200]. Then we get 4 chunks:
   #   chunk (0,0) => shape [50,100], chunk (0,1) => shape [50,100]
   #   chunk (1,0) => shape [50,100], chunk (1,1) => shape [50,100]

   for c_index in manager.iterate_over_chunks():  # yields e.g. (0,0), (0,1), etc.
       c_mask = manager.get_chunk_mask(c_index)   # returns [slice(...), slice(...)]
       # Do partial reads or writes using c_mask, e.g.:
       # manager.FIELDS["some_field"][tuple(c_mask)] = ...

Here’s how chunking typically impacts model usage:

- **Performing chunk-wise computations**: If you want to compute a new field or transform an existing field in
  memory-limited environments, you can iterate over each chunk, load it, do the operation, and write back.
  The method :py:meth:`~pisces.models.grids.base.ModelGridManager.set_in_chunks` automates part of this logic by applying a user-defined
  function chunk-by-chunk.

- **Partial Reading**: If only a slice of the domain is needed (e.g., a cross-section at a certain x value), chunking
  ensures that the HDF5 library reads only those chunks that overlap with your slice, skipping irrelevant data.

.. note::

   The chunk shape must evenly divide the total :py:attr:`GRID_SHAPE`. This is essential so that each chunk is
   the same size, simplifying I/O and iteration logic. Pisces does **not** support partial or irregular chunking
   (like a final truncated chunk on the right-hand boundary).

.. tip::

   If you are always performing entire-grid calculations (like a global integral over the domain), you might
   set ``chunk_shape = grid_shape`` so there’s exactly one chunk. This avoids extra indexing overhead. On the
   other hand, if the domain is so large you can’t fit it in memory, a smaller chunk shape (like ``[64, 64, 64]``)
   could be beneficial.

By default, chunk-based iteration is done axis by axis in integer steps. You can retrieve or transform chunk indexes
with convenience methods like :py:meth:`~pisces.models.grids.base.ModelGridManager.get_chunk_bbox` (to see physical boundaries) or
:py:meth:`~pisces.models.grids.base.ModelGridManager.get_chunk_mask` (to see the slices that define the chunk’s position in array space).

**Summary**:
Chunking is an optional but powerful feature. If your models remain comfortably within your RAM budget, you might never
explicitly handle chunk iteration. But for large-scale runs—or HPC batch processing—chunking is the key to efficient
partial I/O and memory usage.

Field Collections
-----------------
The grid manager includes a :py:class:`~pisces.models.grids.base.ModelFieldContainer` (accessible as
:py:attr:`~pisces.models.grids.base.ModelGridManager.FIELDS`) that organizes all fields in the HDF5 file. This provides:

- Lazy loading: no field data is fetched until it’s sliced.
- A dictionary-like API for creating, copying, or removing fields.
- On-demand indexing for partial data reads.

.. admonition:: Two layers of lazy loading
   :class: hint

   1. **Field-level**: A :py:class:`~pisces.models.grids.base.ModelField` is mostly metadata, pulling actual array data from HDF5 on slice access.
   2. **Manager-level**: The manager only initializes the field container upon request (and each field upon reference),
      so unnecessary data remains uninstantiated.
