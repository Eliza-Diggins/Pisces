"""
Core classes for interacting with particle data in Pisces.

The pisces infrastructure for particle data centers around the :py:class:`ParticleDataset`, which is effectively an
HDF5 wrapper that stores particle data in a hierarchical fashion so that each particle type has its own group and each
field for a particle type has its own dataset within that group.

These dataset objects are designed for light-weight export to other systems, including ``yt``.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import unyt
from numpy.typing import ArrayLike

from pisces.io.hdf5 import HDF5_File_Handle
from pisces.utilities.unit_utils import ensure_ytarray

if TYPE_CHECKING:
    from yt.frontends.stream.data_structures import StreamParticlesDataset


class ParticleDataset:
    """
    A structured container for managing particle data in an HDF5-backed format. This class forms the
    backbone of all particle data management in Pisces.

    The :py:class:`ParticleDataset` class provides a hierarchical storage system where each
    particle type (species) is stored as a **group**, and each field for a species is
    stored as a dataset within that group. This structure allows efficient access,
    manipulation, and export of particle data.

    The dataset is designed for compatibility with external tools, including `yt`,
    allowing for seamless integration into astrophysical and cosmological simulations.

    Examples
    --------

    To create a dataset at a new path, simply call the class constructor and
    feed the desired path. By default, this will load the path if it already exists. To
    overwrite data, simply use ``overwrite=True``.

    >>> # Import the ParticleDataset class from pisces.
    >>> from pisces.particles.base import ParticleDataset
    >>>
    >>> # Create an empty particle dataset at `test.hdf5`.
    >>> pset = ParticleDataset('test.hdf5',overwrite=True)
    >>> print(pset)
    <ParticleDataset: test.hdf5>

    :py:class:`ParticleDataset` instances contain individual HDF5 groups, each corresponding to a
    particle type (species). Species can be added, removed, and altered. When creating a species,
    the number of particles in the file must be specified.

    >>> from pisces.particles.base import ParticleDataset
    >>> pset = ParticleDataset('test.hdf5',overwrite=True)
    >>> print(pset.species)
    {}
    >>>
    >>> # Add a dark matter species with 10^7 particles.
    >>> _ = pset.add_species('dark_matter',int(1e7))
    >>> print(pset.species)
    {'dark_matter': ParticleSpecies(fields=[])}

    Each species (:py:class:`ParticleSpecies`) acts as a container of :py:class:`ParticleField` instances.

    """

    DEFAULT_BUFFER_SIZE: int = 1024
    """ int: The default size of the particle load buffer.
    The buffer size determines the maximum number of particles which are loaded into memory at any given
    time during processes involving particle manipulation. This can be set explicitly when generating
    a :py:class:`ParticleDataset`; however, this value will act as default if it is not specified.
    """

    # @@ LOADING METHODS @@ #
    # These methods form the core of the loading procedures for
    # particle datasets. Subclasses may overwrite these where necessary; however,
    # it is generally possible to leave the core of __init__ alone.
    def __init__(
        self, path: Union[str, Path], buffer_size: int = None, overwrite: bool = False
    ):
        """
        Initialize the particle dataset from disk.

        Parameters
        ----------
        path : Union[str, Path]
            The file path to the HDF5 dataset.

            If ``path`` is an existing path, then an attempt will be made to load the file as a particle dataset. If
            the ``path`` does not exist, then an empty dataset will be created. The ``overwrite`` argument can be used
            to change this behavior to force existing data to be deleted.
        overwrite: bool
            If ``True``, force the creation of a new dataset even if the file already exists.
        buffer_size : int, optional
            The default buffer size (number of particles) to load into memory during processing.
            Defaults to :py:attr:`DEFAULT_BUFFER_SIZE` if not specified.


        Raises
        ------
        ValueError
            If the specified file path does not exist and cannot be created.

        Notes
        -----
        For this class, then ``__init__`` procedure is quite simple. The core logic for setup is passed
        off to the :py:meth:`ParticleDataset.build_skeleton`, which determines if the skeleton already exists and
        returns the buffer. The class then creates a species buffer and starts loading species.
        """
        # Enforce type constraints on self._path and then pass off to the skeleton builder. This
        # will produce the core HDF5 structure before returning the buffer reference.
        self._path = Path(path)
        self._handle = self.build_skeleton(
            path, buffer_size=buffer_size, overwrite=overwrite
        )

        # Set attributes
        self._buffer_size = self._handle.attrs["buffer_size"]

        # Proceed to load substructural information into constituent
        # species and fields.
        self._species = {}
        self._load_particle_species()

    def _load_particle_species(self):
        # Search through all of the groups in the HDF5 file
        # and pull them out as species.
        for k, _ in self._handle.items():
            if k not in self._species:
                self._species[k] = ParticleSpecies(self, k)

    @classmethod
    def build_skeleton(
        cls, path: Union[str, Path], buffer_size: int = None, overwrite: bool = False
    ) -> HDF5_File_Handle:
        """
        Create or initialize an HDF5-backed particle dataset skeleton.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the HDF5 file for the particle dataset. If the file does not exist, it will be created.
        buffer_size : int, optional
            The default buffer size (number of particles) to load into memory during processing.
            Defaults to :py:attr:`DEFAULT_BUFFER_SIZE` if not specified.
        overwrite : bool, optional
            If ``True``, overwrites any existing dataset file. Defaults to ``False``.

        Returns
        -------
        :py:class:`~pisces.io.hdf5.HDF5_File_Handle`
            A new :py:class:`~pisces.io.hdf5.HDF5_File_Handle` instance initialized at the specified path.

        Raises
        ------
        ValueError
            If the file already exists and ``overwrite`` is ``False``.
        """
        # Convert to the Path standard type and validate the input path. Determine
        # if we can return or need to generate a new skeleton.
        path = Path(path)
        if path.exists() and not overwrite:
            return HDF5_File_Handle(path, mode="r+")
        elif path.exists() and overwrite:
            path.unlink()

        # Ensure the file is created or overwritten
        handle = HDF5_File_Handle(path, mode="w")
        handle.attrs["buffer_size"] = buffer_size or cls.DEFAULT_BUFFER_SIZE

        return handle.switch_mode("r+")

    # @@ DUNDER METHODS @@ #
    # These are standard dunder methods for particle classes.
    # They SHOULD NOT BE ALTERED to ensure consistent behavior
    # between subclasses.
    def __str__(self):
        return f"<{self.__class__.__name__}: {self._path}>"

    def __repr__(self):
        return f"{self.__class__.__name__}(species={list(self.species.keys())})"

    def __len__(self) -> int:
        return len(self.species)

    def __getitem__(self, key: str) -> "ParticleSpecies":
        try:
            return self._species[key]
        except KeyError:
            raise KeyError(f"No particle species: {key} for in {self}.")

    def __delitem__(self, key: str) -> None:
        self.remove_species(key)

    def __del__(self):
        self.handle.close()

    def __contains__(self, key: str) -> bool:
        return key in self._species

    def __iter__(self):
        return self.species.__iter__()

    # @@ Core Methods @@ #
    def add_species(
        self, name: str, num_particles: int, overwrite: bool = False
    ) -> "ParticleSpecies":
        """
        Add a new particle species to the dataset. Each species represents a unique particle type / type of massive
        particle. Each species in a particle dataset has its own set of fields carry particle data.

        Parameters
        ----------
        name : str
            The name of the new species. If ``name`` is a duplicate of an existing species, then the behavior will
            depend on ``overwrite``. If ``overwrite`` is ``True``, then the existing species will be deleted, otherwise
            an error is raised.

        num_particles : int
            The number of particles to allocate for the new species.

            .. warning::

                Once a species has been created, the number of particles cannot easily be altered. This is intended
                to prevent particle species from having many fields of different or incomplete length; however, it should
                be noted here because it is necessary to get the number of particles right before initializing the species.

        overwrite : bool, optional
            If ``True``, overwrite existing species. Defaults to ``False``.

        Returns
        -------
        ParticleSpecies
            The newly created particle species.
        """
        ParticleSpecies.build_skeleton(self.handle, name, num_particles, overwrite)
        return self.species[name]

    def remove_species(self, name: str):
        """
        Remove a particle species from the dataset.

        .. warning::

            This will also remove the particle species from the underlying HDF5 dataset. Proceed with caution!

        Parameters
        ----------
        name: str
            The name of the species to remove.
        """
        if name not in self._handle:
            raise ValueError(
                f"No particle species named '{name}' exists in the dataset."
            )
        del self._handle[name]
        self._handle.flush()  # <- ensure that we actually enforce the change.

    def add_index_field(self, species_order: List[str] = None):
        """
        Add an index field for each particle species in the dataset. By adding an index field to the
        particle species in a dataset, each particle gains a unique identity. This is often a necessary component
        of particle based initial conditions.

        Parameters
        ----------
        species_order: List[str], optional
            A list of the species to add indices for and in which order the index should be added.

            .. tip::

                For many simulation codes, there is a specific, underlying order to the particle species. As such,
                the particle indices must proceed in that order as well. This method uses the ``species_order`` argument
                to ensure that there is sufficient control over the order of the species to be sufficiently capable
                for particle based initial conditions formats.

        Returns
        -------
        None

        Notes
        -----
        This method will create a ``"particle_index"`` field in each of the species in the dataset (or those in ``species_order``).
        Each index field is simply an ascending array of integers which is contiguous between species.
        """
        if species_order is None:
            species_order = list(self.species.keys())

        offset = 0
        for species_name in species_order:
            self.species[species_name].add_index_field(offset=offset)
            offset += len(self.species[species_name])

    def to_yt(
        self,
        length_units: str = "pc",
        mass_units: str = "Msun",
        time_units: str = "s",
        velocity_units: str = "km/s",
        magnetic_units: str = "T",
        bbox: ArrayLike = None,
        **kwargs,
    ) -> "StreamParticlesDataset":
        """
        Convert the particle dataset to a ``yt`` dataset. See :py:func:`yt.loaders.load_particles`.

        Parameters
        ----------
        length_units: str, optional
            The units of the length of the particles. Defaults to ``"pc"``.
        mass_units: str, optional
            The units of the mass of the particles. Defaults to ``"Msun"``.
        time_units: str, optional
            The units of the time of the particles. Defaults to ``"s"``.
        velocity_units: str, optional
            The units of the velocity of the particles. Defaults to ``"km/s"``.
        magnetic_units: str, optional
            The units of the magnetic field of the particles. Defaults to ``"T"``.
        bbox: ArrayLike, optional
            The bounding box of the particles. Defaults to ``None``, which will lead the
            code to determine a minimal bounding box holding all of the particles.
        kwargs:
            Additional arguments to pass to :py:func:`yt.loaders.load_particles`.

        Returns
        -------
        :py:class:`~yt.frontends.stream.data_structures.StreamParticlesDataset`
        """
        # Begin with importing yt. If we fail, we want to raise a
        # wrapped error because yt is not explicitly a dependency.
        try:
            import yt
        except ImportError as ex:
            raise ImportError(
                "Cannot convert particle dataset to yt because yt cannot be imported."
            ) from ex

        # Validation / setup steps.
        # In order to generate the particle dataset, the bounding box has to be generated
        # and the units need to be coerced.
        _particle_fields = (
            {}
        )  # This is where we store all the data for in-memory loading.

        unyt.unit_systems.UnitSystem(
            "psystem", length_units, mass_units, time_units, "K"
        )

        # Determine the bounding box for the particle dataset.
        # We do this by iterating through all of the particles unless a bbox is specified.
        _bbox = bbox
        if _bbox is None:
            # Set the bbox by iterating through the particles.
            _bbox = np.array([[0, 1] for _ in range(3)])
            for _, species in self.species.items():
                position_data = species.FIELDS["particle_position"][...].to_value(
                    length_units
                )
                ndim = position_data.shape[-1]
                _bbox = np.array(
                    [
                        [
                            np.amin([np.amin(position_data[:, _i]), _bbox[_i, 0]]),
                            np.amax([np.amax(position_data[:, _i]), _bbox[_i, 1]]),
                        ]
                        for _i in range(ndim)
                    ]
                )

        # Add all of the particle fields to the dataset.
        for ptype, species in self.species.items():
            for field, field_data in species.FIELDS.items():
                if field not in ["particle_position", "particle_velocity"]:
                    _particle_fields[(ptype, field)] = field_data[...]
                elif field == "particle_velocity":
                    # Correct the velocity
                    for i, ax_name in enumerate(["x", "y", "z"]):
                        _particle_fields[(ptype, field + f"_{ax_name}")] = field_data[
                            :, i
                        ].to_value(velocity_units)
                elif field == "particle_position":
                    # Correct the velocity
                    for i, ax_name in enumerate(["x", "y", "z"]):
                        _particle_fields[(ptype, field + f"_{ax_name}")] = field_data[
                            :, i
                        ].to_value(length_units)

            # Check and add in particle velocities if necessary
            if "particle_velocity" not in species.FIELDS:
                for _, ax_name in enumerate(["x", "y", "z"]):
                    _particle_fields[
                        (ptype, "particle_velocity" + f"_{ax_name}")
                    ] = unyt.unyt_array(
                        np.zeros(species.num_particles), "km/s"
                    ).to_value(
                        velocity_units
                    )

        # noinspection PyTypeChecker
        return yt.load_particles(
            _particle_fields,
            bbox=_bbox,
            length_unit=length_units,
            mass_unit=mass_units,
            velocity_unit=velocity_units,
            time_unit=time_units,
            magnetic_unit=magnetic_units,
            **kwargs,
        )

    # @@ Core Properties @@ #
    @property
    def path(self) -> Path:
        """
        The file path associated with this :py:class:`ParticleDataset` instance's location on disk.

        Returns
        -------
        Path:
            The path to the underlying file.
        """
        return self._path

    @property
    def handle(self) -> HDF5_File_Handle:
        """
        Access the HDF5 file handle associated with the particle dataset.

        This handle provides a direct interface to the underlying HDF5 file, enabling
        interaction with raw data stored in the file.

        Returns
        -------
        :py:class:`~pisces.io.hdf5.HDF5_File_Handle`
            The HDF5 file handle for the model.

        Notes
        -----
        The handle is used internally for reading and writing data to the file.
        Direct interaction with the handle should be avoided unless necessary.
        """
        return self._handle

    @property
    def species(self) -> Dict[str, "ParticleSpecies"]:
        """
        The particle species present in this dataset. Each particle species represents a unique
        particle type which, in turn, has its own associated set of fields carrying particle data.

        Returns
        -------
        dict of str, :py:class:`ParticleSpecies`
            The dictionary of the particle species.
        """
        self._load_particle_species()
        return self._species

    @property
    def buffer_size(self) -> int:
        """
        The buffer size for this :py:class:`ParticleDataset` instance. The buffer size is the
        largest array of particles to load into data at once; which helps to ensure memory safety
        when working with large datasets.

        Returns
        -------
        int
        The buffer size for this :py:class:`ParticleDataset` instance.
        """
        return self._buffer_size

    @property
    def num_particles(self) -> Dict[str, int]:
        """
        The number of particles present in this :py:class:`ParticleDataset` instance.

        Returns
        -------
        dict of str, int
            Dictionary containing key-value pairs corresponding to each particle species and the number
            of particles present in the dataset of that type.
        """
        return {k: self.species[k].num_particles for k in self.species.keys()}


class ParticleSpecies:
    """
    Container class representing a particular type of particle in a
    :py:class:`ParticleDataset` instance.
    """

    # @@ LOADING METHODS @@ #
    # These methods form the core of the loading procedures for
    # particle species dataset. Subclasses may overwrite these where necessary; however,
    # it is generally possible to leave the core of __init__ alone.
    def __init__(self, particle_dataset: ParticleDataset, name: str):
        """
        Load the :py:class:`ParticleSpecies` from an existing :py:class:`ParticleDataset` instance given
        the ``name`` of the particle type.

        Parameters
        ----------
        particle_dataset: :py:class:`ParticleDataset`
            The particle dataset containing this particle type and its constituent data.
        name: str
            The name of the particle type.

        Raises
        ------
        ValueError
            if ``name`` is not a group in the ``particle_dataset``.
        """
        # Validate existence of the correct species group within the
        # particle dataset itself.
        if name not in particle_dataset.handle:
            raise ValueError(f"Particle dataset {name} not found in HDF5 file.")

        # Create basic attributes
        self._name = name
        self._dataset = particle_dataset
        self._handle = self._dataset.handle[name]

        # Load the fields.
        self._fields = {}
        self._load_particle_fields()

    def _load_particle_fields(self):
        """
        Load all available particle fields for this species from the HDF5 file.

        This method iterates over the datasets stored in the species group in the HDF5 file
        and registers them as :py:class:`ParticleField` instances.

        Returns
        -------
        None
        """
        # Search through all of the groups in the HDF5 file
        # and pull them out as species.
        for k, _ in self._handle.items():
            if k not in self._fields:
                self._fields[k] = ParticleField(self, k)

    @classmethod
    def build_skeleton(
        cls, handle: h5py.Group, name: str, num_particles: int, overwrite=None
    ):
        """
        Create or retrieve an HDF5 group for a particle species.

        If the species already exists, it is returned unless ``overwrite=True`` is specified,
        in which case the existing group is deleted and a new one is created.

        Parameters
        ----------
        handle : h5py.Group
            The HDF5 file handle where the species is being created.
        name : str
            The name of the species to be created.
        num_particles : int
            The number of particles in this species.
        overwrite : bool, optional
            If ``True``, deletes any existing species with the same name before creating a new one.

        Returns
        -------
        h5py.Group
            The HDF5 group associated with the species.
        """
        # Check for an existing dataset and manage it if necessary. Use
        # overwrite to determine behavior.
        if name in handle:
            if overwrite:
                del handle[name]
            else:
                return handle

        # We no longer have the species present in the handle. Generate a new one.
        # The group can be empty except for specifying the num_particles attribute.
        species_group = handle.require_group(name)
        species_group.attrs["num_particles"] = num_particles

        return species_group

    # @@ DUNDER METHODS @@ #
    # These are standard dunder methods for particle classes.
    # They SHOULD NOT BE ALTERED to ensure consistent behavior
    # between subclasses.
    def __str__(self):
        return f"<{self.__class__.__name__}: {self._name}>"

    def __repr__(self):
        return f"{self.__class__.__name__}(fields={list(self.FIELDS.keys())})"

    def __len__(self) -> int:
        return self.num_particles

    def __getitem__(self, key: str) -> "ParticleField":
        try:
            return self.FIELDS[key]
        except KeyError:
            raise KeyError(f"No particle field: {key} for in {self}.")

    def __delitem__(self, key: str) -> None:
        self.remove_field(key)

    def __contains__(self, key: str) -> bool:
        return key in self.FIELDS

    def __iter__(self):
        return self.FIELDS.__iter__()

    # @@ Utility Methods @@ #
    def get_chunk_slices(self) -> List[slice]:
        """
        Generate a list of slices representing chunks of particles.

        This method divides the particle species into chunks based on the buffer size of the dataset.
        Each chunk is represented as a slice, ensuring efficient processing of particles in manageable
        portions.

        Returns
        -------
        List[slice]
            A list of slices, where each slice represents a chunk of particles.

        Notes
        -----
        - The buffer size is defined in the parent `ParticleDataset` instance.
        - The last chunk may contain fewer particles than the buffer size, depending on the total number of particles.
        """
        # Determine the number of full chunks and initialize the list of slices
        n_chunks = (self.num_particles // self.dataset.buffer_size) + 1

        # Generate slices for each full chunk
        chunk_slices = [
            slice(
                chunk_id * self.dataset.buffer_size,
                (chunk_id + 1) * self.dataset.buffer_size,
            )
            for chunk_id in range(n_chunks - 1)
        ]

        # Add the final slice for the remaining particles
        chunk_slices.append(
            slice((n_chunks - 1) * self.dataset.buffer_size, self.num_particles)
        )

        return chunk_slices

    # @@ Core Methods @@ #
    def add_field(
        self,
        name: str,
        /,
        element_shape: Optional[Tuple[int, ...]] = None,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        *,
        overwrite: bool = False,
        dtype: str = "f8",
        units: str = "",
    ):
        """
        Add a new field to the particle species.

        This method creates a new dataset in the HDF5 file under this species, storing per-particle
        data such as position, velocity, mass, or any other property.

        Parameters
        ----------
        name : str
            The name of the new field.
        element_shape : tuple of int, optional
            The shape of individual elements in the field (e.g., ``(3,)`` for a 3D vector).
        data : unyt.unyt_array or np.ndarray, optional
            Initial data for the field. If provided, its shape must match the expected dimensions.
        overwrite : bool, optional
            If ``True``, allows overwriting an existing field with the same name.
        dtype : str, optional
            The data type of the field (default is ``float64``).
        units : str, optional
            The physical units of the field (e.g., ``"kpc"``, ``"Msun/kpc**3"``).

        Returns
        -------
        ParticleField
            The newly created field.
        """
        field = ParticleField(
            self,
            name,
            element_shape=element_shape,
            data=data,
            dtype=dtype,
            units=units,
            overwrite=overwrite,
        )

        return field

    def remove_field(self, name: str):
        """
        Remove a field from the species.

        This method deletes the specified field from the HDF5 dataset and the in-memory
        representation of the species.

        Parameters
        ----------
        name : str
            The name of the field to remove.

        Raises
        ------
        ValueError
            If the field does not exist in this species.
        """
        if name not in self.FIELDS:
            raise ValueError(
                f"No particle species named '{name}' exists in the dataset."
            )
        del self._handle[name]

    def add_index_field(self, offset: int = 0, overwrite: bool = False):
        """
        Add an index field to uniquely identify each particle.

        The index field assigns a unique integer ID to each particle, useful for tracking
        particles across time steps or different datasets.

        Parameters
        ----------
        offset : int, optional
            The starting index value for this species. Defaults to ``0``.
        overwrite : bool, optional
            If ``True``, allows overwriting an existing index field.

        Returns
        -------
        None
        """
        self.add_field(
            "particle_index",
            data=np.arange(self.num_particles) + offset,
            dtype="int",
            units="",
            overwrite=overwrite,
        )

    def add_offsets(
        self,
        position_offset: Optional[unyt.unyt_array] = None,
        velocity_offset: Optional[unyt.unyt_array] = None,
    ) -> None:
        """
        Add offsets to the particle positions and/or velocities.

        This method applies specified offsets to the particle positions and velocities in chunks,
        ensuring efficient memory usage for large datasets.

        Parameters
        ----------
        position_offset : unyt.unyt_array, optional
            Offset to apply to the particle positions. Must have compatible units with
            the `particle_position` field. If `None`, no position offset is applied.
        velocity_offset : unyt.unyt_array, optional
            Offset to apply to the particle velocities. Must have compatible units with
            the `particle_velocity` field. If `None`, no velocity offset is applied.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the required `particle_position` or `particle_velocity` fields are missing.
            If the units of the provided offsets are incompatible with the field units.

        Notes
        -----
        - This method processes particles in chunks to minimize memory usage.
        - Unit consistency is enforced using the `unyt` library.
        """
        # Get the list of slices for chunked processing
        chunks = self.get_chunk_slices()

        # Apply position offset
        if position_offset is not None:
            if "particle_position" not in self.FIELDS:
                raise ValueError(
                    f"No position field (`particle_position`) in {self}. Cannot add offsets."
                )
            position_offset = ensure_ytarray(
                position_offset, self.FIELDS["particle_position"].units
            )

            for chunk in chunks:
                self.FIELDS["particle_position"][chunk] += position_offset

        # Apply velocity offset
        if velocity_offset is not None:
            if "particle_velocity" not in self.FIELDS:
                raise ValueError(
                    f"No velocity field (`particle_velocity`) in {self}. Cannot add offsets."
                )
            velocity_offset = ensure_ytarray(
                velocity_offset, self.FIELDS["particle_velocity"].units
            )

            for chunk in chunks:
                self.FIELDS["particle_velocity"][chunk] += velocity_offset

    @property
    def FIELDS(self) -> Dict[str, "ParticleField"]:
        """
        Get the dictionary of fields associated with this particle species.

        Returns
        -------
        dict of str to ParticleField
            A dictionary where keys are field names and values are :py:class:`ParticleField` instances.
        """
        self._load_particle_fields()
        return self._fields

    @property
    def handle(self) -> h5py.Group:
        """
        Get the HDF5 group associated with this particle species.

        This property provides direct access to the HDF5 group in which this species' data is stored.
        It allows interaction with the underlying file structure for reading or modifying particle fields.

        Returns
        -------
        h5py.Group
            The HDF5 group corresponding to this species.
        """
        return self._handle

    @property
    def num_particles(self) -> int:
        """
        Get the total number of particles in this species.

        The number of particles is fixed when the species is created and is stored as an attribute
        in the HDF5 dataset.

        Returns
        -------
        int
            The total number of particles in this species.
        """
        return self._handle.attrs["num_particles"]

    @property
    def dataset(self) -> ParticleDataset:
        """
        Get the parent :py:class:`ParticleDataset` instance that this species belongs to.

        This allows access to the dataset that contains this species, enabling operations
        that involve multiple species or dataset-wide modifications.

        Returns
        -------
        ParticleDataset
            The parent dataset containing this species.
        """
        return self._dataset


class ParticleField(unyt.unyt_array):
    """
    Represents a data field associated with a particular particle species.

    The `ParticleField` class extends `unyt.unyt_array` to provide integration with HDF5-backed storage,
    enabling efficient management and manipulation of particle data. Each instance corresponds to a field
    within a particle species dataset, supporting unit-aware operations and on-disk slicing.

    Notes
    -----
    - This class is designed to operate seamlessly with HDF5-backed particle datasets.
    - It supports unit-aware arithmetic operations using the `unyt` library.
    - Slicing operations interact directly with the on-disk HDF5 dataset for efficiency.
    """

    def __new__(
        cls,
        species_dataset: ParticleSpecies,
        name: str,
        /,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        element_shape: Optional[Tuple[int, ...]] = None,
        *,
        overwrite: bool = False,
        dtype: str = "f8",
        units: str = "",
    ):
        # Build or retrieve the underlying HDF5 dataset
        dataset = cls.build_skeleton(
            species_dataset,
            name,
            element_shape=element_shape,
            data=data,
            overwrite=overwrite,
            dtype=dtype,
            units=units,
        )

        # Extract final units from the dataset attributes
        _units = dataset.attrs["units"]

        # Create the unyt array instance with placeholder data
        obj = super().__new__(cls, [], units=_units)

        # Attach references to the constructed object
        obj._name = name
        obj.units = unyt.Unit(_units)
        obj.dtype = dataset.dtype
        obj.buffer = dataset

        return obj

    # noinspection PyUnusedLocal
    def __init__(
        self,
        species_dataset: ParticleSpecies,
        name: str,
        /,
        element_shape: Optional[Tuple[int, ...]] = None,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        *,
        overwrite: bool = False,
        dtype: str = "f8",
        units: str = "",
    ):
        """
        Initialize the ParticleField.

        Parameters
        ----------
        species_dataset : ParticleSpecies
            The dataset representing the particle species.
        name : str
            The name of the field.
        data : Optional[Union[unyt.unyt_array, np.ndarray]], optional
            Initial data for the field.
        overwrite : bool, optional
            If `True`, overwrites an existing field. Defaults to `False`.
        dtype : str, optional
            Data type for the field. Defaults to "f8".
        units : str, optional
            Units of the field. Defaults to an empty string.
        """
        pass

    @classmethod
    def build_skeleton(
        cls,
        species_dataset: ParticleSpecies,
        name: str,
        /,
        element_shape: Optional[Tuple[int, ...]] = None,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        *,
        overwrite: bool = False,
        dtype: str = "f8",
        units: str = None,
    ) -> h5py.Dataset:
        """
        Create or retrieve the HDF5 dataset for the field.

        Parameters
        ----------
        species_dataset : ParticleSpecies
            The parent dataset containing all the fields for the particle species.
        name : str
            The name of the field.
        element_shape : Optional[Tuple[int, ...]], optional
            The shape of each element in the dataset. Defaults to a scalar.
        data : Optional[Union[unyt.unyt_array, np.ndarray]], optional
            Initial data for the field. If provided, it must match the shape.
        overwrite : bool, optional
            If `True`, overwrites an existing field. Defaults to `False`.
        dtype : str, optional
            The data type for the field. Defaults to "f8".
        units : str, optional
            Units of the field. If `data` is a `unyt_array`, its units are used.

        Returns
        -------
        h5py.Dataset
            The HDF5 dataset for the field.

        Raises
        ------
        ValueError
            If the data shape or units are incompatible.
        """
        handle = species_dataset.handle
        if overwrite and name in handle:
            del handle[name]

        if name in handle:
            return handle[name]

        if element_shape is None:
            element_shape = tuple()

        n_particles = species_dataset.num_particles
        shape = (n_particles, *element_shape)

        if data is not None:
            if not np.array_equal(data.shape, shape):
                raise ValueError(f"Expected shape {shape}, but got {data.shape}.")
            if isinstance(data, unyt.unyt_array):
                data_units = data.units
                if units and units != str(data_units):
                    data = data.to(units).value
                else:
                    units = data_units
            data = np.asarray(data, dtype=dtype)
        else:
            units = units or ""

        dataset = handle.create_dataset(name, shape=shape, data=data, dtype=dtype)
        dataset.attrs["units"] = str(units)
        return dataset

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override numpy's ufunc behavior to ensure unit consistency.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The universal function to apply.
        method : str
            The method of the ufunc.
        *inputs : tuple
            Input arrays.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        unyt_array
            Result of the operation with units.
        """
        cast_inputs = tuple(
            (x.view(unyt.unyt_array) if isinstance(x, ParticleField) else x)
            for x in inputs
        )
        result = getattr(ufunc, method)(*cast_inputs, **kwargs)
        if isinstance(result, unyt.unyt_array):
            result.units = self.units
        return result

    def __setitem__(
        self, key: Union[slice, int], value: Union[unyt.unyt_array, np.ndarray]
    ):
        """
        Set a slice of the field's data.

        Parameters
        ----------
        key : slice or int
            The indices to set.
        value : unyt.unyt_array or np.ndarray
            The values to set.

        Raises
        ------
        ValueError
            If units are incompatible.
        """
        if isinstance(value, np.ndarray):
            value = unyt.unyt_array(value, self.units)
        elif not isinstance(value, unyt.unyt_array):
            raise ValueError("Value must be unyt_array or numpy.ndarray.")

        value = value.to_value(self.units)
        self.buffer[key] = value

    def __getitem__(self, key: Union[slice, int]) -> unyt.unyt_array:
        """
        Retrieve a slice of the field's data.

        Parameters
        ----------
        key : slice or int
            The indices to retrieve.

        Returns
        -------
        unyt.unyt_array
            The requested slice with units.
        """
        arr = self.buffer[key]
        return unyt.unyt_array(arr, units=self.units)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.buffer.shape
