"""
Particle sampling methods and classes for models.

"""
from typing import Tuple, Optional, Any, TYPE_CHECKING, Type, Dict, List, Callable
import numpy as np
from collections import OrderedDict
if TYPE_CHECKING:
    from pisces.models.base import Model

class _SamplerMeta(type):
    def __new__(
        mcs: Type["_SamplerMeta"],
        name: str,
        bases: Tuple[Type, ...],
        clsdict: Dict[str, Any]
    ) -> Type[Any]:
        r"""
        Create a new class object, search through the metadata looking for any samplers that can be
        inherited.

        Parameters
        ----------
        mcs : type
            The metaclass (this class).
        name : str
            The name of the class being created (e.g., "MyModel").
        bases : tuple
            The base classes for this new class.
        clsdict : dict
            The class attributes namespace (methods, class variables, etc.).

        """
        # Create the base class object without adulteration from the
        # metaclass.
        cls = super().__new__(mcs, name, bases, clsdict)

        # If the class is not an abstract base class, we need to construct the pathways
        # dictionary. Otherwise, we leave the entire dictionary blank.
        cls._SAMPLERS = OrderedDict({})
        mcs.construct_samplers(cls, clsdict)

        # Sort the samplers
        cls._SAMPLERS = OrderedDict(sorted(cls._SAMPLERS.items(), key=lambda t: t[1]['priority']))

        return cls

    @staticmethod
    def construct_samplers(cls: '_SamplerMeta', clsdict: Dict[str, Any]) -> None:
        # Search either the entire class dictionary or the entire MRO depending on the
        # specifying flag in the base code. Look for any registered attributes and ensure
        # that they get added to the sampler dictionary with the correct restrictions.
        _priorities_seen = set() # The set of priorities we've seen.
        seen_attrs = set()
        inherits_samplers = clsdict.get("_INHERITS_SAMPLERS", False)
        if inherits_samplers:
            bases = cls.__mro__
        else:
            bases = [cls]

        for base in bases:
            # The built-in `object` or other sentinel classes won't contain relevant info. We
            # can skip over them.
            if base is object:
                continue

            # Explore the base's __dict__ to find candidate methods/attributes.
            for attr_name, method_obj in base.__dict__.items():
                # Make sure we don't re-process the same (base, attr_name) combination.
                if (base, attr_name) in seen_attrs:
                    continue
                seen_attrs.add((base, attr_name))

                # Detect the _is_sampler attribute to mark a method as
                # needing to be registered as a sampler.
                if not getattr(method_obj, "_is_sampler", False):
                    continue

                # All of the methods that make it to this point are
                # actual samplers. We want to simply register them to the
                # dictionary now. We need to pull out the relevant metadata
                # from the method.
                _sampler_name = getattr(method_obj, "_sampler_name")
                _sampler_checker = getattr(method_obj, "_sampler_checker", None)
                _sampler_priority = getattr(method_obj, "_sampler_priority", 0)
                _priorities_seen.add(_sampler_priority)

                if _sampler_name not in cls._SAMPLERS:
                    cls._SAMPLERS[_sampler_name] = {
                        "checker": _sampler_checker,
                        "priority": _sampler_priority,
                        "method": attr_name,
                    }
                else:
                    raise ValueError(f"Duplicate sampler: {_sampler_name}")

class ModelSampler(metaclass=_SamplerMeta):
    """
    Base class for model samplers.

    Each :py:class:`~pisces.models.base.Model` has a connected :py:class:`ModelSampler` class
    which controls how the user is able to sample from the model once it is generated.
    """
    # @@ Flags @@ #
    # The flags here are used by the meta class to dictate behavior. They
    # can be modified in subclass implementations as needed.
    _INHERITS_SAMPLERS: bool = True

    # @@ Class Variables @@ #
    # These variables act as defaults for the sampler behavior and can
    # be modified to change the behavior of the class.
    DEFAULT_FIELD_PARTICLE_TYPE_MAP: Dict[str, str] = {}

    # @@ Class Construction / Setup @@ #
    # These methods control construction and setup. They are
    # constructed modularly, so there is little reason to alter __init__,
    # but other methods in the process may be changed in subclasses.
    def __init__(self,
                 model: 'Model'):
        # Link the model to the instance and check to ensure that this is a valid
        # sampler type for the provided model.
        self._model = self._setup_model(model)

        # Setup the particle -- field matched list.
        self._field_particle_dict = self.__class__.DEFAULT_FIELD_PARTICLE_TYPE_MAP.copy()


    def _setup_model(self,model: 'Model') -> 'Model':
        # Validate that the model recognizes this class as
        # its sampler.
        if type(self) != model.SAMPLE_TYPE:
            raise ValueError(f"Model class {model.__class__.__name__} expects sampler class {model.SAMPLE_TYPE}.")

        # Return the model so it can be set.
        return model

    # @@ Dunder Methods @@ #
    # These should NOT be altered in subclasses.
    def __str__(self):
        return f"<{self.__class__.__name__} of {self.model}>"

    def __repr__(self):
        return self.__str__()

    # @@ General Methods @@ #
    def add_field_particle_match(self, field_name: str, particle_species: str, overwrite: bool = False):
        """
        Add a match between a field name and a particle species.

        Parameters
        ----------
        field_name : str
            The name of the field.
        particle_species : str
            The particle species to match with the field.
        overwrite : bool, optional
            If True, overwrite any existing match for the field. Defaults to False.

        Raises
        ------
        ValueError
            If the field already has a match and overwrite is False.
        """
        if field_name in self._field_particle_dict and not overwrite:
            raise ValueError(f"Field '{field_name}' is already matched to "
                             f"particle '{self._field_particle_dict[field_name]}'. "
                             f"Set overwrite=True to overwrite.")
        self._field_particle_dict[field_name] = particle_species

    def remove_field_particle_match(self, field_name: str):
        """
        Remove the match for a given field name.

        Parameters
        ----------
        field_name : str
            The name of the field.

        Raises
        ------
        KeyError
            If the field name is not in the dictionary.
        """
        if field_name not in self._field_particle_dict:
            raise KeyError(f"Field '{field_name}' is not matched to any particle.")
        del self._field_particle_dict[field_name]

    def get_particle_species(self, field_name: str) -> Optional[str]:
        """
        Get the particle species matched to a given field.

        Parameters
        ----------
        field_name : str
            The name of the field.

        Returns
        -------
        Optional[str]
            The particle species matched to the field, or None if no match exists.
        """
        return self._field_particle_dict.get(field_name, None)

    def get_field_names(self, particle_species: str) -> List[str]:
        """
        Get a list of fields matched to a given particle species.

        Parameters
        ----------
        particle_species : str
            The particle species.

        Returns
        -------
        List[str]
            A list of field names matched to the particle species.
        """
        return [field for field, particle in self._field_particle_dict.items() if particle == particle_species]

    # @@ Properties @@ #
    # These properties can be added to, but properties should not
    # be removed to ensure consistent behavior for users.
    @property
    def model(self) -> 'Model':
        """
        The :py:class:`~pisces.models.base.Model` instance this sampler belongs to.
        """
        return self._model

    @property
    def samplers(self) -> Dict[str, Any]:
        return self._SAMPLERS[:]

    @property
    def field_particle_dict(self) -> Dict[str, str]:
        """
        Get a copy of the current field-to-particle dictionary.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping field names to particle species.
        """
        return self._field_particle_dict.copy()

    # @@ Sampling Methods @@ #
    # The sampling methods are the core of this class. They should be added
    # to / altered in subclasses to produce the necessary structure to support
    # a particular model.
    #
    # Generally, subclasses should not remove a method without good reason.
    def _lookup_sampler(self, field_name: str) -> Tuple[str, Callable]:
        """
        Look up the appropriate sampler for a given field name.

        Parameters
        ----------
        field_name : str
            The name of the field for which a sampler is needed.

        Returns
        -------
        str
            The name of the sampler method.

        Notes
        -----
        If no sampler explicitly matches the field, this method returns 'default_sampler'.
        """
        # Iterate through each of the samplers in our
        # registry. They should already be ordered by priority.
        for sampler_name, sampler_meta in self._SAMPLERS.items():
            if sampler_meta["checker"]:
                checker = getattr(self, sampler_meta["checker"],None)
                if checker is None:
                    return sampler_name, getattr(self, sampler_meta['method'])

                if checker(field_name):
                    return sampler_name, getattr(self,sampler_meta['method'])

        return "default", getattr(self, 'default_sampler')


    def sample_positions(self,
                         particle_species: str,
                         num_particles: int,
                         /,
                         *,
                         sample_from: Optional[str] = None,
                         use_sampler: Optional[str] = None,
                         **kwargs):
        # Perform the validation procedures for the inputs.
        # The sample_from parameter needs to be set to determine what field is being sampled from and
        # the sampler needs to be identified.
        if sample_from is None:
            # look up the field in our dictionary for fields.
            if particle_species in self._field_particle_dict:
                sample_from = self._field_particle_dict[particle_species]
            else:
                raise ValueError(f"No particle species '{particle_species}' found in {self}; failed to find a field to"
                                 f" sample from.\nIf a field should be available, try specifying it manually with `sample_from=`")

        if sample_from not in self.model.FIELDS:
            raise ValueError(f"Cannot sample particle positions from field {sample_from} because model {self.model} has no such field.")

        # Determine if they specified a sampler. If not, we need to look it up.
        if use_sampler is None:
            use_sampler = self._lookup_sampler(sample_from)
        else:
            # validate the sampler that was provided. It still needs to pass validation
            # and actually exist.
            if use_sampler not in self.samplers:
                raise ValueError(f"The sampler '{use_sampler}' does not exist in {self}.")

            _checker = self.samplers[use_sampler]["checker"]
            if _checker and not getattr(self,_checker)(sample_from):
                raise ValueError(f"The sampler `{use_sampler}` failed its validation for field {sample_from}.")


        # Prepare to perform the sampling procedure.




def sampler(name: str, checker: Optional[str] = None, priority: int = 0):
    """
    Decorator to register a method as a sampler in the `ModelSampler` class.

    Parameters
    ----------
    name : str
        The name of the sampler method. This is how the sampler will be referred to
        in the registry.
    checker : Optional[str], optional
        A string reference to another class method which will be called to check this
        sampler. By default, this is None, which will make this sampler capable of
        running in all cases.
    priority : int, optional
        The priority of the sampler. Samplers with higher priority values are
        considered before those with lower values. Defaults to 0.

    Returns
    -------
    Callable
        The decorated method, now registered as a sampler.
    """
    def decorator(func):
        # Attach metadata to the function
        setattr(func, "_is_sampler", True)
        setattr(func, "_sampler_name", name)
        setattr(func, "_sampler_checker", checker)
        setattr(func, "_sampler_priority", priority)

        return func

    return decorator

if __name__ == '__main__':
    print(ModelSampler(None).samplers)

