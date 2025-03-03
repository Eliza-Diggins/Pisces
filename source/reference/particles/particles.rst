.. _particles:

=========================
Particles in Pisces
=========================
For any Pisces model, it is (in principle) possible to convert the grid-based :py:class:`~pisces.models.base.Model` instance
to a particle-based :py:class:`~pisces.particles.base.ParticleDataset` instance which represents the same physical system.
Doing so can have many advantages for certain computations and for working with external tools. Because particles are relatively
simple data structures, it is easy to migrate particle data to / from pisces from / to other tools. In this reference, we'll
describe how to work with particles in Pisces and how to generate them.

Overview
--------

Fundamentally, a system in Pisces may be thought of as a collection of different types (**species**) of particles. Each particle
has a position in space and a value for one or many **fields**, which determine the physical properties of the particle. In Pisces,
these particle systems are represented by the :py:class:`~pisces.particles.base.ParticleDataset`, which provides an interface between
Pisces and an underlying HDF5 file for storing the particle data.

Each :py:class:`~pisces.particles.base.ParticleDataset` is, itself, an HDF5 file with a number of **groups** each representing
a single particle **species** (:py:class:`~pisces.particles.base.ParticleSpecies`). Each **species** group can then also have a number
of **fields** (:py:class:`~pisces.particles.base.ParticleField`).

Fields are just HDF5 datasets containing a value for each particle of a particular type.

.. _opening_viewing_particle_dataset:
Opening / Viewing Particle Dataset
''''''''''''''''''''''''''''''''''

Opening a particle dataset file as a :py:class:`~pisces.particles.base.ParticleDataset` is simple; simply call the class
on the particular filename you want to open.

.. code-block:: python

    >>> from pisces.particles import ParticleDataset
    >>> pd = ParticleDataset('particles.hdf5')

If the file exists, then it will be opened as a particle dataset. If it does not, then a new dataset is generated with
no particles.

Within a :py:class:`~pisces.particles.base.ParticleDataset`, particle data are divided into species, each represented
by a :py:class:`~pisces.particles.base.ParticleSpecies`. Once the dataset is initialized, you can see which species
are present by examining the species attribute or by indexing the dataset directly:

.. code-block:: python

    >>> pd.species
    {'dark_matter': ParticleSpecies(fields=['particle_position', 'particle_velocity', ...]),
     'gas': ParticleSpecies(fields=['particle_position', 'density', ...])}

    >>> dm = pd["dark_matter"]
    >>> print(dm)
    <ParticleSpecies: dark_matter>

Each :py:class:`~pisces.particles.base.ParticleSpecies` holds various fields, which are arrays of data (in HDF5)
for every particle in that species. You can see them via:

.. code-block:: python

    >>> list(dm.FIELDS.keys())
    ['particle_position', 'particle_velocity', 'mass', ...]

You can retrieve a field just like dictionary indexing:

.. code-block:: python

    >>> positions = dm["particle_position"]
    >>> print(positions.shape)
    (1000000, 3)

For large datasets, Pisces handles these fields as on-disk arrays. Slicing them will load only a portion of data
into memory, helping to manage large particle counts efficiently.

.. _adding_and_removing_particles:
Adding / Removing Species and Fields
''''''''''''''''''''''''''''''''''''

You can add a new species by calling :py:meth:`~pisces.particles.base.ParticleDataset.add_species`. This requires
specifying a unique species name and the total number of particles in that species:

.. code-block:: python

    >>> from pisces.particles.base import ParticleDataset
    >>> pds = ParticleDataset("particles.hdf5")
    >>> new_species = pds.add_species("stars", num_particles=1000)

.. note::

    The total number of particles in a species is fixed when created to avoid inconsistencies in field lengths.

If you need to remove an entire species (and all of its fields), you can call
:py:meth:`~pisces.particles.base.ParticleDataset.remove_species`:

.. code-block::

    >>> pds.remove_species("stars")

.. warning::

    Be aware that this will remove the species data permanently from the underlying HDF5 file.

Each species can have multiple fields, such as ``particle_position``, ``particle_velocity``, ``mass``, ``density``, etc.
You can add a new field to a species with :py:meth:`~pisces.particles.base.ParticleSpecies.add_field`:

.. code-block:: python

    >>> # Create some dummy data for the new field
    >>> import numpy as np
    >>> import unyt

    >>> density_data = (np.random.rand(1000) * 1e-4) * unyt.Unit("Msun/kpc**3")
    >>> pds["stars"].add_field("density", data=density_data, units="Msun/kpc**3")

In this example, the new field ``density`` is created for the ``stars`` species, storing a per-particle density array.
By providing a ``unyt.unyt_array`` (or specifying ``units=...``), you ensure that unit consistency is tracked.

.. note::

    The shape of the data must match the number of particles (and any element shape, if applicable).

If you already have a field by this name and want to replace it, set ``overwrite=True``:

.. code-block:: python

    >>> pds["stars"].add_field("density", data=new_density_data, units="Msun/kpc**3", overwrite=True)

You can remove a field from a species (and delete it from the HDF5 file) via
:py:meth:`~pisces.particles.base.ParticleSpecies.remove_field`:

.. code-block:: python

    >>> pds["stars"].remove_field("density")

This completely removes the "density" dataset from the HDF5 file, so proceed with caution if you still need the data.

Generating Particles From Models
--------------------------------
While it is sometimes useful to be able to generate particle datasets from scratch, the most important use case for
:py:class:`~pisces.particles.base.ParticleDataset` is modeling existing :py:class:`~pisces.models.base.Model` instances.
To do this, we need to generate particles from the model; a somewhat non-trivial endeavour. In general, there are 3 steps
to this sampling process:

1. **Sampling**: Given that the model contains density fields of different types, generate particles which have the same
   distribution as prescribed by the model.
2. **Interpolate**: For each particle generated by sampling, assign field values to the particles based on the values of
   the model's fields.
3. **Virialize**: For collisionless particles, assign particle velocities which stabilize the distribution dynamically.

.. note::

    In general, the first and third of these steps is quite difficult. Users interested in the details of each of these
    processes should read :ref:`sampling` and :ref:`virialization`.

To manage these processes, Pisces links together the :py:class:`~pisces.models.base.Model` and the :py:class:`~pisces.particles.base.ParticleDataset`
with a **linking-class** called :py:class:`~pisces.models.virialize.Virializer`.

Overview
''''''''

The :py:class:`~pisces.models.virialize.Virializer` is designed to provide a link between models and particles. Because different
types of models require different logic for particle production, subclasses of :py:class:`~pisces.models.virialize.Virializer` are
written for specific :py:class:`pisces.models.base.Model` subtypes.

.. note::

    For developers, these "custom" virializers are usually housed in the corresponding model's module.

To link a :py:class:`~pisces.models.virialize.Virializer` to a :py:class:`~pisces.models.base.Model`, all you have to do
is initialize the virializer and give it the model to process and the filename into which the particle dataset should be
written:

.. code-block:: python

    >>> from pisces.models.galaxy_clusters.models import ClusterModel
    >>> from pisces.models.galaxy_clusters.virializers import SphericalClusterVirializer
    >>> model = ClusterModel('model.hdf5')
    >>> vir = SphericalClusterVirializer(model,"particles.hdf5")

Upon initializing the :py:class:`~pisces.models.virialize.Virializer`, an empty :py:class:`~pisces.particles.base.ParticleDataset`
is created corresponding to the specified file:

.. code-block:: python

    >>> print(vir.particles)
    <ParticleDataset: particles.hdf5>

You can also access the model:

.. code-block:: python

    >>> print(vir.model)
    <ClusterModel: 'model.hdf5'>

Mapping From Model to Particles
'''''''''''''''''''''''''''''''
A small portion of the logic in the :py:class:`~pisces.models.virialize.Virializer` is dedicated to providing mappings
between fields and particles and the equivalent data in the model being converted. All virializer classes have **at least**
two lookup tables:

- :py:attr:`~pisces.models.virialize.Virializer.field_lut`: is a dictionary which tells the virializer what fields a
  specific particle type should get and what it should be called on the particle side. For example,

  .. code-block:: python

      >>> print(vir.field_lut)
      {
      'dark_matter': {'density':'dark_matter_density'},
      'gas': {'density': 'gas_density',
             'internal_energy': 'temperature'}
      }

  corresponds to a virializer which has particle types ``dark_matter`` and ``gas`` and renames ``temperature`` (model) to
  ``internal_energy`` (particle dataset).

  .. note::

      There are 5 special fields which are **always** added to the output particle dataset despite not being
      specified in the lookup table:

      - ``particle_position`` and ``particle_position_native``: The coordinates of the particle (cartesian and native).
      - ``particle_velocity`` and ``particle_velocity_native``: The velocity of the particle (catesian and native).
      - ``particle_mass``: The mass of the particle.

- :py:attr:`~pisces.models.virialize.Virializer.density_lut`: is a dictionary pointing each particle type to the density
  field (in the model) from which it should be sampled. This is necessary for the virializer to be able to accurately generate
  particles.

You can access and manipulate these lookup tables as needed in order to modify the behavior of the default virializer before
creating the resulting particle dataset.

Sampling
''''''''
The first critical function of the :py:class:`~pisces.models.virialize.Virializer` is to produce particles which have the correct
distribution to match the density specified in the model. This is the **sampling** process and is performed by calling the
:py:meth:`~pisces.models.virialize.Virializer.generate_particles` method.

The :py:meth:`~pisces.models.virialize.Virializer.generate_particles` method takes a single argument, ``num_particles``, which
is a ``dict`` specifying the number of particles of each type to generate:

.. code-block:: python

    >>> vir.generate_particles({'gas':1_000_000,'dark_matter':1_000_000})
    Pisces : [INFO     ] 2025-03-02 16:35:10,342 Sampling positions for species 'gas' (1000000 particles).
    Pisces : [INFO     ] 2025-03-02 16:35:10,490 Sampling positions for species 'dark_matter' (1000000 particles).
    Pisces : [INFO     ] 2025-03-02 16:35:10,626 Completed particle position sampling.

.. hint::

    Under the hood, the virializer uses the :py:attr:`~pisces.models.virialize.Virializer.density_lut` to find the
    corresponding density field in the model and then samples from it. This is all managed in the private method ``_sample_particles``,
    which is modified in subclasses to implement the correct sampling methodology.

Interpolating
'''''''''''''

Once the particles have been generated, its easy enough to map different fields onto them. This is performed using the
:py:meth:`~pisces.models.virialize.Virializer.interpolate_fields` method, which will use the sampled positions for each
of the particles to determine the correct field values.

.. code-block:: python

    >>> vir.interpolate_fields()
    Pisces : [INFO     ] 2025-03-02 16:40:47,876 Interpolating: temperature -> gas,internal_energy.
    Pisces : [INFO     ] 2025-03-02 16:40:48,325 Interpolating: gas_density -> gas,density.
    Pisces : [INFO     ] 2025-03-02 16:40:48,755 Interpolating: pressure -> gas,pressure.
    Pisces : [INFO     ] 2025-03-02 16:40:49,184 Interpolating: gravitational_potential -> gas,gravitational_potential.
    Pisces : [INFO     ] 2025-03-02 16:40:49,619 Interpolating: dark_matter_density -> dark_matter,density.
    Pisces : [INFO     ] 2025-03-02 16:40:49,976 Interpolating: gravitational_potential -> dark_matter,gravitational_potential.

Virializing
'''''''''''

.. warning::

    Not yet Implemented

Developing Custom Virializers
'''''''''''''''''''''''''''''

The :py:class:`~pisces.models.virialize.Virializer` provides the core structure for sampling particles, interpolating fields, and
(when implemented) virializing velocities. However, different types of astrophysical models require different logic for generating
particle distributions. To address this, developers can subclass :py:class:`~pisces.models.virialize.Virializer` to implement
custom virialization strategies tailored to specific :py:class:`~pisces.models.base.Model` subclasses.

A custom virializer is implemented as a subclass of :py:class:`~pisces.models.virialize.Virializer`,
typically housed within the same module as the corresponding :py:class:`~pisces.models.base.Model`.

.. code-block:: python

    from pisces.models.virialize import Virializer
    from pisces.models.base import Model

    class MyCustomVirializer(Virializer):
        _VALID_PARTICLE_TYPES = ['gas', 'dark_matter']
        DEFAULT_DENSITY_FIELD_LUT = {
            'gas': 'gas_density',
            'dark_matter': 'dark_matter_density'
        }
        DEFAULT_FIELD_LUT = {
            'gas': {'density': 'gas_density', 'temperature': 'temperature'},
            'dark_matter': {'density': 'dark_matter_density'}
        }

This subclass defines:
    - ``_VALID_PARTICLE_TYPES``: The particle species the virializer supports.
    - ``DEFAULT_DENSITY_FIELD_LUT``: A dictionary mapping each particle type to the corresponding density field in the model.
    - ``DEFAULT_FIELD_LUT``: A dictionary specifying which fields should be interpolated onto each particle type.

Adding Model Validation
"""""""""""""""""""""""

Each virializer should ensure that the provided model is compatible. To do this, override the
:py:meth:`~pisces.models.virialize.Virializer._validate_model` method:

.. code-block:: python

    class MyCustomVirializer(Virializer):
        def _validate_model(self, model: Model) -> None:
            if not hasattr(model, 'FIELDS'):
                raise ValueError("Model must have a 'FIELDS' attribute containing field mappings.")

This ensures that any model passed to the virializer has the required properties.

Implementing Custom Particle Sampling
"""""""""""""""""""""""""""""""""""""

The most critical step in writing a custom virializer is defining how particles are sampled from the model.
This is handled by the :py:meth:`~pisces.models.virialize.Virializer._sample_particles` method, which must be implemented in each subclass.

.. code-block:: python

    import numpy as np
    import unyt

    class MyCustomVirializer(Virializer):
        def _sample_particles(self, species: str, num_particles: int) -> unyt.unyt_quantity:
            """
            Custom implementation for sampling particle positions.

            Parameters
            ----------
            species : str
                The particle species being sampled.
            num_particles : int
                The number of particles to generate.

            Returns
            -------
            unyt.unyt_quantity
                The mass of each sampled particle.

            Notes
            -----
            This example samples positions randomly in a unit cube.
            """
            # Generate random positions in a unit cube
            self.particles[species]["particle_position_native"][:] = np.random.rand(num_particles, 3)

            # Assign equal mass to all particles
            mass = unyt.unyt_quantity(1.0, "Msun") / num_particles
            return mass

This method:
    - Generates random positions in a unit cube (for illustration).
    - Assigns equal mass to all particles.

Developers should replace this logic with a physically meaningful sampling strategy.

Custom Field Interpolation
""""""""""""""""""""""""""

To implement a custom interpolation method, override :py:meth:`~pisces.models.virialize.Virializer._interpolate_field`:

.. code-block:: python

    class MyCustomVirializer(Virializer):
        def _interpolate_field(self, species: str, model_field: str, particle_field: str, **kwargs):
            """
            Custom field interpolation method.

            Parameters
            ----------
            species : str
                The particle species whose positions we use for interpolation.
            model_field : str
                The field in the model to interpolate.
            particle_field : str
                The corresponding field in the particle dataset.

            Notes
            -----
            This method assigns a constant value for demonstration purposes.
            """
            self.particles[species][particle_field][:] = np.full(
                self.particles[species][particle_field].shape, 100.0
            )

This trivial example assigns a constant value to all particles, but in a real implementation, it should interpolate
values from the model grid onto the particles.

Using a Custom Virializer
"""""""""""""""""""""""""

Once implemented, the new virializer can be used like any other:

.. code-block:: python

    >>> from pisces.models.galaxy_clusters.models import ClusterModel
    >>> model = ClusterModel('model.hdf5')
    >>> vir = MyCustomVirializer(model, "custom_particles.hdf5")

    >>> vir.generate_particles({'gas': 1_000_000, 'dark_matter': 500_000})
    >>> vir.interpolate_fields()

Manipulating Particle Datasets
------------------------------

Altering a Single Particle
''''''''''''''''''''''''''

Combining Particle Datasets
'''''''''''''''''''''''''''

Truncating Particle Datasets
''''''''''''''''''''''''''''

3rd Party Compatibility
-----------------------
