"""
Virialization manager classes for converting models to particle datasets.

The virialization toolkit provides :py:class:`Virializer` classes, which are connected to specific
:py:class:`~pisces.models.base.Model` subclasses to permit conversion between models and particle datasets.
"""
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

import numpy as np
import unyt
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pisces.particles.base import ParticleDataset
from pisces.utilities.config import pisces_params
from pisces.utilities.logging import devlog, mylog

if TYPE_CHECKING:
    from pisces.models.base import Model


class VirializerMeta(ABCMeta):
    """
    A metaclass to validate that DEFAULT_DENSITY_FIELD_LUT and DEFAULT_FIELD_LUT
    are consistent with _VALID_PARTICLE_TYPES whenever a Virializer subclass is defined.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        # Create the new class object first
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # If this is the base or abstract class, or if _VALID_PARTICLE_TYPES is None,
        # we skip validation. You can modify this check to suit your needs.
        if ABC in bases or not getattr(cls, "_VALID_PARTICLE_TYPES", None):
            return cls

        # Check DEFAULT_DENSITY_FIELD_LUT
        if getattr(cls, "DEFAULT_DENSITY_FIELD_LUT", None) is not None:
            for density_key in cls.DEFAULT_DENSITY_FIELD_LUT:
                if density_key not in cls._VALID_PARTICLE_TYPES:
                    raise ValueError(
                        f"Class {cls.__name__}: '{density_key}' in DEFAULT_DENSITY_FIELD_LUT "
                        "is not in _VALID_PARTICLE_TYPES."
                    )

        # Check DEFAULT_FIELD_LUT
        if getattr(cls, "DEFAULT_FIELD_LUT", None) is not None:
            for field_key in cls.DEFAULT_FIELD_LUT:
                if field_key not in cls._VALID_PARTICLE_TYPES:
                    raise ValueError(
                        f"Class {cls.__name__}: '{field_key}' in DEFAULT_FIELD_LUT "
                        "is not in _VALID_PARTICLE_TYPES."
                    )

        return cls


class Virializer(ABC, metaclass=VirializerMeta):
    """
    Abstract base class for all descendant virializers. This class defines the core logic
    for sampling particles, fields, and viralizing models.

    The :py:class:`Virializer` class manages three core actions:

    1. (Sampling) Sampling particles from :py:class:`~pisces.models.base.Model` instances.
    2. (Interpolating) Interpolating fields from a model onto created particles.
    3. (Virializing) Virialize the particle velocities.

    Each subclass should achieve these three actions; however, the methodology by which the sampling
    and virialization step are accomplished will (in general) vary significantly between models depending
    on the relevant physics and mathematical properties of the systems.
    """

    # @@ CLASS VARIABLES @@
    # These variables should be modified in subclasses of the generic
    # virializer to encode the detection of particle types and fields.
    _VALID_PARTICLE_TYPES: List[str] = None
    """ list of str: The list of particle types recognized by this class.

    Each valid particle type should be included here. If a particle type is not recognized, errors are
    raised when an action is performed that refers to the unknown particle type.

    Notes
    -----
    When developing subclasses, developers should simply include all of the particles that are relevant to
    the connected model. For standard particle types (i.e. AREPO recognized particles), standard naming conventions
    should be followed. For more esoteric cases, novel naming conventions can be used.
    """
    _PARTICLE_DATASET_TYPE: Type[ParticleDataset] = ParticleDataset
    """ Type: The type of particle dataset to use for saving / referencing particles.

    In general, this can be left as the default (:py:class:`~pisces.particles.base.ParticleDataset`); however,
    in some cases, developers may have a custom particle dataset class into which the virializer should generate particles.
    """
    DEFAULT_DENSITY_FIELD_LUT: Dict[str, str] = None
    """ dict of str,str: Lookup table to determine which density field is linked to each particle type.
    Each particle type should be specified by a ``key`` and each ``value`` should be a corresponding
    string referencing the density field from which those particles should be sampled from in the model.

    Notes
    -----
    When developing new subclasses, this should be completed to ensure that the :py:class:`Virializer` knows which
    model fields are used to construct the sampling apparatus. Not all particle types specified in ``_VALID_PARTICLE_TYPES``
    need to be specified here; however, all of the ``key`` values in :py:attr:`DEFAULT_DENSITY_FIELD_LUT` must be
    present in ``_VALID_PARTICLE_TYPES``. If a particle type does not have a matching density field or the density
    field cannot be found, then a warning / error is raised depending on context.
    """
    DEFAULT_FIELD_LUT: Dict[str, Dict[str, str]] = None
    """ dict of str, Dict[str,str]: Lookup table to indicate which fields are included for each species.
    Each element (key,value pair) of :py:attr:`DEFAULT_FIELD_LUT` corresponds to a particular particle type and
    another map specifying all of the fields that should exist for that particle type. The ``values`` of the
    dictionary should, themselves, be a dictionary containing the names of the fields to create for that particle type as
    keys and the corresponding names of the fields in the model from which to interpolate as values.

    Notes
    -----
    Like :py:attr:`DEFAULT_DENSITY_FIELD_LUT`, missing particle types will raise a warning / error depending on context;
    however, if an invalid particle type is included then an error is raised.

    There are an additional 4 fields which are always present in :py:attr:`DEFAULT_FIELD_LUT`:

    - ``particle_position_native`` and ``particle_velocity_native`` specify the position and velocity vectors in the native
      coordinate system of the model.
    - ``particle_position`` and ``particle_velocity`` specify the position and velocity vectors in cartesian coordinates.

    These should **NOT** be included in the :py:attr:`DEFAULT_FIELD_LUT`.
    """

    def __init__(
        self,
        model: "Model",
        path: Union[str, Path],
        overwrite: bool = False,
        pset_kw: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the :py:class:`Virializer` instance.

        This constructor sets up the core components needed for virializing a model:
        validating the provided model, configuring output paths, and initializing the
        particle dataset. It also supports subclassing by allowing keyword arguments
        for fine-grained control over dataset creation and other behaviors.

        Parameters
        ----------
        model : :py:class:`~pisces.models.base.Model`
            The simulation model from which to sample the particles. This must be
            an instance of a subclass of :py:class:`~pisces.models.base.Model`, and it
            will be validated before being assigned to the virializer.
        path : Union[str, Path]
            File path for saving the sampled particle data. This should point to an
            HDF5 file or another appropriate storage format used by the particle dataset.
        overwrite : bool, optional
            If ``True``, an existing dataset at the specified path will be deleted before
            creating a new one. If ``False``, an error will be raised if a dataset already exists.
        pset_kw : dict, optional
            Additional keyword arguments passed to the constructor of
            :py:class:`pisces.particles.base.ParticleDataset`. These options allow
            subclass developers to customize dataset creation (e.g., buffer size, chunking strategy).
        kwargs : dict
            Additional keyword arguments that can be used by subclasses to introduce
            custom behavior. These are stored in ``self._kwargs`` and can be used by various
            processes executed by the class. Subclasses should specifically document the relevant keyword arguments.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified path exists and `overwrite` is `False`.

        Notes
        -----

        .. rubric:: Developer Notes

        This constructor follows a structured initialization process to ensure
        consistency across different virializer implementations. Below is an outline
        of the key steps, along with considerations for subclass development:

        **1. Path Validation and Handling Overwrites**

            - The `path` parameter is immediately converted to a `Path` object.
            - If the file already exists and `overwrite=False`, a `ValueError` is raised.
            - If `overwrite=True`, the existing file is deleted before continuing.

            🔒 **Subclass Restriction**: This logic **cannot** be modified in subclasses,
            ensuring that all virializer instances maintain the same overwrite behavior.

        **2. Model Registration and Validation**

            - The provided `model` is stored as `self._model`.
            - The `_validate_model` method is called to ensure the model is properly structured.

            🔧 **Subclass Customization**: `_validate_model` can be overridden to introduce
            additional model validation checks specific to a subclass.

        **3. Particle Dataset Initialization**

            - The class attribute `_PARTICLE_DATASET_TYPE` is used to determine which
              dataset type should be instantiated (default: :py:class:`ParticleDataset`).
            - The `pset_kw` dictionary allows developers to pass optional parameters
              to modify dataset behavior (e.g., setting buffer sizes, storage format optimizations).

            🔧 **Subclass Customization**: By modifying `_PARTICLE_DATASET_TYPE`, a subclass
            can change the particle dataset class used. Additionally, subclasses can modify `pset_kw`
            dynamically before calling `super().__init__()`.

        **4. Instance-Specific Lookup Tables**

            - The look-up tables for density fields (`_density_field_lut`) and field mappings (`_field_lut`)
              are copied from their class-level counterparts.
            - These instance-level copies allow modifications without affecting other instances.

            🔧 **Subclass Customization**: Subclasses may modify these tables **after** calling
            `super().__init__()` to introduce different field mappings or density references.

        **5. General Keyword Storage**

            - Any additional keyword arguments provided in `kwargs` are stored in `self._kwargs`.
            - This allows flexible configuration in subclasses without requiring modification
              of the base class constructor.

        """
        # Validate the user-provided path and ensure that it is compliant with the
        # specified overwriting settings. Raise an error if any issues appear. This segment of
        # the initialization procedure cannot be altered in subclasses.
        self._path = Path(path)
        if self.path.exists() and not overwrite:
            raise ValueError(
                f"The particle dataset at {path} already exists. To overwrite it use overwrite=True."
            )
        elif self.path.exists() and overwrite:
            self.path.unlink()

        # Register the sampling model and validate it. The validation step can be altered in
        # subclasses to customize behavior.
        self._model: "Model" = model
        self._validate_model(self._model)

        # Create the particle dataset. We permit a variety of kwargs to be passed.
        pset_kw = {} if pset_kw is None else pset_kw
        self._particles = self.__class__._PARTICLE_DATASET_TYPE(self.path, **pset_kw)

        # Generate the lookup tables connected to the instance. These are then
        # the standard reference for manipulating the permitted fields and density fields.
        self._density_field_lut = self.__class__.DEFAULT_DENSITY_FIELD_LUT.copy()
        self._field_lut = self.__class__.DEFAULT_FIELD_LUT.copy()

        # Create the keywords reference
        self._kwargs = kwargs

    @abstractmethod
    def _validate_model(self, model: "Model") -> None:
        """
        Initialization sub-process which confirms that the model being passed is valid for this
        :py:class:`Virializer` class. By default, this method simply passes and there are no checks
        on the model.

        Parameters
        ----------
        model : :py:class:`~pisces.models.base.Model`
            The simulation model from which to sample the particles.

        Returns
        -------
        None
        """
        pass

    # @@ PARTICLE GENERATION @@ #
    # All of the methods in this part of the class pertain to
    # managing sampling procedures.
    def generate_particles(
        self, num_particles: dict[str, int], overwrite: bool = False, **kwargs
    ):
        """
        Generate and store particles in the particle dataset based on the specified species and counts.

        This method performs the following sequence of operations:

        3. **Particle Initialization**:

            - Allocates space for each species and creates necessary fields.

        4. **Particle Sampling**:

            - Calls `_sample_particles` to assign particle positions based on the density field.

        5. **Coordinate Transformation**:

            - Converts native model coordinates to Cartesian coordinates.

        Parameters
        ----------
        num_particles : dict of str -> int
            A dictionary mapping particle species names (e.g., `"gas"`, `"dark_matter"`) to
            the number of particles to generate for each species.
        overwrite : bool, optional
            If ``True``, existing particles of the same species will be removed and replaced.
            If ``False``, an error is raised if the species already exists. Default is ``False``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a requested particle species is not in `_VALID_PARTICLE_TYPES`.
        ValueError
            If a requested species lacks a valid density field in the lookup table.
        ValueError
            If an existing particle species is found and `overwrite=False`.

        Notes
        -----
        .. rubric:: Developer Notes

        - The **density lookup table (`self.density_lut`)** is used to verify each species.
        - **Subclasses must implement `_sample_particles`**, which determines how particle positions are sampled.
        - The coordinate transformation step assumes that **`self.model.coordinate_system.to_cartesian`**
          correctly maps native coordinates to Cartesian space.
        - **Logging behavior**:
            - Warnings are issued when species are skipped (e.g., zero particles, missing fields).
            - Informational logs indicate overwrites and sampling progress.

        """
        # Validate the requested particle types / counts. We need to verify that all of the particle
        # types are valid and that they each have a density lookup table match.
        # Check the number of particles and return if we eliminate species.
        _removed_species = []
        for ptype, pcount in num_particles.items():
            # Check that we didn't get zero for the particle counts.
            if pcount == 0:
                _removed_species.append(ptype)
                mylog.warning(
                    "Skipping %s because there are no particles to generate.", ptype
                )
                continue

            # Check that ptype is at least a valid particle type.
            if ptype not in self._VALID_PARTICLE_TYPES:
                raise ValueError(
                    f"'{ptype}' is not a valid particle type. Valid types: {self._VALID_PARTICLE_TYPES}"
                )

            # Check that we can find a match in the density lut.
            if self.density_lut.get(ptype, None) is None:
                raise ValueError(
                    f"'{ptype}' has no match in the density field lookup table. Cannot identify a sampling density field."
                )
            elif self.density_lut[ptype] not in self.model.FIELDS:
                _removed_species.append(ptype)
                mylog.warning(
                    "Skipping %s because it's density field (%s) is not present in the model.",
                    ptype,
                    self.density_lut[ptype],
                )
                continue

            # Manage overwriting issues
            if (ptype in self.particles.species) & (~overwrite):
                raise ValueError(
                    f"The particle species {ptype} already exists and overwrite is False."
                )
            elif (ptype in self.particles.species) & overwrite:
                # We need to remove and reinitialize this particle species group.
                mylog.info("Removing and replacing particle species %s.", ptype)
                self.particles.remove_species(ptype)

            # Initialize the new species with the correct number of
            # particles and other information.
            self.particles.add_species(ptype, pcount, overwrite=False)

        # Remove the species that got eliminated and return none if all of the
        # species had to be removed.
        num_particles = {
            k: v for k, v in num_particles.items() if k not in _removed_species
        }
        if not len(num_particles):
            return

        # We have the skeleton for each of them - now we just need to
        # sample the particle positions for each one.
        _mass_unit = kwargs.pop("mass_units", "Msun")
        with logging_redirect_tqdm(loggers=[self.model.logger, devlog, mylog]):
            for species in tqdm(
                num_particles.keys(),
                desc="Generating particles",
                unit="species",
                disable=pisces_params["system.preferences.disable_progress_bars"],
            ):
                # Start by generating the particle position field for each of the
                # particle types in the system.
                self.particles[species].add_field(
                    "particle_position_native",
                    element_shape=(self.model.coordinate_system.NDIM,),
                    dtype="f8",
                    units="",
                )
                self.particles[species].add_field(
                    "particle_position",
                    element_shape=(self.model.coordinate_system.NDIM,),
                    dtype="f8",
                    units=str(self.model.grid_manager.length_unit),
                )
                # Add the mass field
                self.particles[species].add_field(
                    "particle_mass",
                    dtype="f8",
                    units=_mass_unit,
                )

                # Dispatch the sampling procedure to the dispatcher.
                mylog.info(
                    "Sampling positions for species '%s' (%d particles).",
                    species,
                    num_particles[species],
                )
                mass = self._sample_particles(species, num_particles[species])

                # Set the mass
                self.particles[species]["particle_mass"][:] = mass.to_value(
                    _mass_unit
                ) * np.ones_like(self.particles[species]["particle_mass"][:])

                # Now create the true position
                self.particles[species]["particle_position"][
                    :, :
                ] = self.model.coordinate_system.to_cartesian(
                    self.particles[species]["particle_position_native"][:, :]
                )

        mylog.info("Completed particle position sampling.")

    @abstractmethod
    def _sample_particles(self, species: str, num_particles: int) -> unyt.unyt_quantity:
        pass

    # @@ FIELD INTERPOLATION @@ #
    # These methods handle interpolating fields onto particles
    # from the specified model.
    def interpolate_fields(self, fields: Optional[dict] = None):
        """
        Interpolates model fields onto sampled particles.

        This method assigns values from the model to particles by interpolating
        field data from the model's grid onto particle positions. The method
        supports interpolating **all fields for all species** (default behavior)
        or **a specific subset of fields** for selected species.

        The interpolation process consists of the following steps:

        1. **Field Mapping Construction**:
            - If no specific fields are provided, it uses the default field lookup table (`field_lut`).
            - If fields are specified, it verifies that they exist in both the model and dataset.
        2. **Validation**:
            - Ensures species exist in the particle dataset.
            - Checks if fields are available in the field lookup table and exist in the model.
        3. **Interpolation Execution**:
            - Iterates through species and requested fields.
            - Calls `_interpolate_field` for each valid field.

        Parameters
        ----------
        fields : dict, optional
            A dictionary specifying which fields to interpolate for each species.
            The structure should be:

            .. code-block:: python

                {
                    "species1": ["field1", "field2"],
                    "species2": ["field3", "field4"]
                }

            If `None` (default), **all fields for all particle species in the dataset**
            will be interpolated based on the field lookup table (`field_lut`).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a requested species does not exist in the dataset.
        ValueError
            If a requested field does not exist in the model.

        Notes
        -----
        .. rubric:: Developer Notes

        - This method **only interpolates fields for species that have already been sampled**.
        - It relies on `self.field_lut` to map model fields to particle dataset fields.
        - Uses `_interpolate_field()` for field interpolation.
        - **Warning messages are logged** if species or fields are missing.
        """
        # Perform the validation steps. We need to lookup all of the specified fields, fill them in, etc.
        # and ensure that everything get's setup correctly.
        # Build the fields attribute if it doesn't already exist.
        if fields is None:
            # If the fields kwarg isn't filled, we generate all the fields for all the currently
            # extant particle species in the dataset.
            fields = {
                k: list(v.keys())
                for k, v in self.field_lut.items()
                if k in self.particles.species
            }

        # Check all of the fields and particles to ensure that we can use them.
        _removed_species = {}
        for ptype, pfields in fields.items():
            # Check that the ptype is valid. We don't need to check the valid particles list
            # because the existence of the ptype in the dataset is a stronger condition.
            if ptype not in self.particles.species:
                _removed_species[ptype] = None
                mylog.warning(
                    "Cannot interpolate fields for species %s because it has not been sampled yet.",
                    ptype,
                )
                continue
            else:
                _removed_species[ptype] = []

            # Now check that all of the fields are accessible.
            for _field in pfields:
                _mfield = self.field_lut[ptype].get(_field, None)
                if _mfield is None:
                    _removed_species[ptype].append(_field)
                    mylog.warning(
                        "Cannot interpolate field %s for species %s because it's not in the field LUT.",
                        _field,
                        ptype,
                    )
                    continue
                elif _mfield not in self.model.FIELDS:
                    _removed_species[ptype].append(_field)
                    mylog.warning(
                        "Cannot interpolate field %s for species %s because it's model equivalent (%s) doesn't exist.",
                        _field,
                        ptype,
                        self.density_lut[ptype],
                    )
                    continue

        # Now construct the field map from the un-removed species.
        fields = {k: v for k, v in fields.items() if _removed_species[k] is not None}
        for ptype in fields:
            fields[ptype] = [
                _field
                for _field in fields[ptype]
                if _field not in _removed_species[ptype]
            ]

        if not len(fields):
            return

        # Process each of the specific cases.
        n_fields = np.sum([len(fields[ptype]) for ptype in fields])
        with logging_redirect_tqdm(loggers=[self.model.logger, devlog, mylog]):
            # Setup the progress bar.
            pbar = tqdm(
                desc="Interpolating particle fields",
                unit="fields",
                total=n_fields,
                leave=False,
                disable=pisces_params["system.preferences.disable_progress_bars"],
            )

            for ptype, pfields in fields.items():
                for _field in pfields:
                    _mfield = self.field_lut[ptype][_field]

                    # create the particle field.
                    self.particles[ptype].add_field(
                        _field, units=self.model.FIELDS[_mfield].units
                    )
                    self._interpolate_field(ptype, _mfield, _field)
                    pbar.update(1)

            pbar.close()

        return

    @abstractmethod
    def _interpolate_field(
        self, species: str, model_field: str, particle_field: str, **kwargs
    ):
        """
        Interpolates a field from the simulation model onto the particles of a given species.

        This method constructs an **interpolator** for the specified field from the model’s
        grid, then applies it to determine the corresponding values at the positions of the
        sampled particles. It supports batch processing of particles for efficiency.

        The interpolation process consists of the following steps:

        1. **Interpolator Construction**:
            - Uses `self.model.grid_manager.build_field_interpolator()` to create an interpolator.
            - Supports configurable options via `kwargs`, such as caching and bounds error handling.
        2. **Setup and Preprocessing**:
            - Determines which axes are relevant for interpolation.
            - Identifies whether logarithmic transformations are needed.
        3. **Batch Processing for Efficiency**:
            - Iterates through particles in buffered chunks (to optimize memory usage).
            - Applies logarithmic transformations to coordinates where needed.
            - Uses the interpolator to compute the field values.

        Parameters
        ----------
        species : str
            The name of the particle species whose positions we use for interpolation.
        model_field : str
            The name of the field in the model to interpolate (e.g., `"velocity_x"`).
        particle_field : str
            The name of the corresponding field in the particle dataset (e.g., `"vx"`).
        **kwargs
            Additional options for the interpolator. Includes:
            - `cache` (bool): Whether to cache the interpolator for reuse. Default is `False`.
            - `bounds_error` (bool): If `True`, raises an error when interpolating out of bounds.
              Default is `False` (allows extrapolation).
            - `fill_value` (float or None): Value used for out-of-bounds interpolation.
              Default is `None` (uses extrapolation).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the species or fields are not found in the particle dataset.

        Notes
        -----
        .. rubric:: Developer Notes

        - The **interpolator is constructed with customizable settings** via `kwargs`.
        - **Logarithmic coordinate transformations** are applied where needed.
        - Uses **batch processing** to avoid excessive memory consumption.
        - Calls `self.particles[species]['particle_position_native']` to obtain
          the native coordinate positions of particles.
        """
        # Construct the interpolator for the specified field. The bounds errors need to
        # be turned off so that we have access to out of bounds interpolation.
        field_interpolator = self.model.grid_manager.build_field_interpolator(
            model_field,
            cache=kwargs.pop("cache", False),
            bounds_error=kwargs.pop("bounds_error", False),
            fill_value=kwargs.pop("fill_value", None),
            **kwargs,
        )

        # Setup necessary 1-time use flags and other variables.
        field_axes = self.model.FIELDS[model_field].AXES
        axes_mask = self.model.coordinate_system.build_axes_mask(field_axes)
        log_mask = self.model.grid_manager.is_log_mask[axes_mask]
        buffer_size = self.particles.buffer_size
        num_part = self.particles[species].num_particles

        # Iterate through the particle positions in buffer length groupings and
        # use the interpolator to determine the field values.
        idx = 0
        with logging_redirect_tqdm(loggers=[self.model.logger, devlog, mylog]):
            pbar = tqdm(
                desc=f"Interpolating: {model_field} -> {species}",
                unit="particles",
                disable=pisces_params["system.preferences.disable_progress_bars"],
                total=num_part,
                leave=False,
            )

            mylog.info(f"Interpolating: {model_field} -> {species},{particle_field}.")
            while idx < self.particles[species][particle_field].shape[0]:
                # Determine how many particles to process this round and pull out their
                # coordinates so that we can interpolate them.
                _parts_this_iter = min(num_part - idx, buffer_size)
                _particle_coords = self.particles[species]["particle_position_native"][
                    idx : idx + _parts_this_iter, axes_mask
                ]

                # Correct the coordinates for log spacings and then
                # pass to the interpolator.
                _particle_coords[:, log_mask] = np.log10(_particle_coords[:, log_mask])
                self.particles[species][particle_field][
                    idx : idx + _parts_this_iter
                ] = field_interpolator(
                    _particle_coords,
                )
                idx += _parts_this_iter
                pbar.update(_parts_this_iter)

            pbar.close()

    @property
    def path(self) -> Path:
        """
        The location of the HDF5 particle dataset on disk.

        Returns
        -------
        Path
            The file path where the particle dataset is stored.
        """
        return self._path

    @property
    def model(self):
        """
        The simulation model associated with this sampler.

        Returns
        -------
        Model
            The model used for sampling particle data.
        """
        return self._model

    @property
    def particles(self):
        """
        The particle dataset associated with this sampler.

        This property provides access to the particle dataset where sampled
        particle data is stored, including species and their associated fields.

        Returns
        -------
        ParticleDataset
            The particle dataset instance managed by this sampler.
        """
        return self._particles

    @property
    def field_lut(self):
        """
        Lookup table to indicate which fields are included for each species.
        Each particle type should be specified by a ``key`` and each ``value`` should be a corresponding
        dict of fields from the ``model`` (key) and their name in the particle dataset (value) to interpolate onto the sampled particles.
        """
        return self._field_lut

    @property
    def density_lut(self):
        """
        Lookup table to determine which density field is linked to each particle type.
        Each particle type should be specified by a ``key`` and each ``value`` should be a corresponding
        string referencing the density field from which those particles should be sampled from in the model.

        .. warning::

            If a particle type is not included, then an error is raised if the user attempts to sample that
            particle type from the model.
        """
        return self._density_field_lut


if __name__ == "__main__":
    q = ParticleDataset("test.hdf5")
    print(q)
