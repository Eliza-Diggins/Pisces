"""
Core structures for managing Pisces initial conditions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import numpy as np
from unyt import Unit, unyt_array

if TYPE_CHECKING:
    from pisces.geometry.base import CoordinateSystem
    from pisces.models.base import Model
    from pisces.particles.base import ParticleDataset


@dataclass
class _ModelReference:
    """
    Class referencing a specific model in an initial conditions system.
    """

    model: "Model"
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray = np.asarray([0, 0, 1])


class InitialConditions(ABC):
    """
    Generic class for managing initial conditions in Pisces.
    """

    # @@ CLASS ATTRIBUTES @@ #
    # These should be altered in subclasses to meet the standard unit conventions / other
    # behaviors of the code base. New class attributes can be added in subclasses to
    # increase the versatility of the class. These are required in all subclasses.
    #
    # -- Coordinate System Attributes -- #
    COORDINATE_SYSTEM: Type["CoordinateSystem"] = None
    COORDINATE_SYSTEM_PARAMETERS: Optional[Dict[str, Any]] = None
    # -- Unit System Attributes -- #
    DEFAULT_LENGTH_UNIT: str = "kpc"
    DEFAULT_VELOCITY_UNIT: str = "km/s"

    @abstractmethod
    def _initialize_coordinates(self, *args, **kwargs) -> "CoordinateSystem":
        """
        Initializes the coordinate system for the initial conditions. This abstract method
        can (and should) be altered in subclasses to ensure that the coordinate system is configured
        as needed to interface with the code base used for the simulation.

        Parameters
        ----------
        args :
            The arguments passed to ``__init__``.
        kwargs :
            The keyword arguments passed to ``__init__``.

        Returns
        -------
        CoordinateSystem
            The coordinate system for the initial conditions.

        Notes
        -----
        By default, this method simply initializes the coordinate system using the coordinate system
        parameters attribute. Alterations could allow user input on the coordinate system specification or
        other more advanced behaviors.
        """
        # Construct the coordinate system parameters.
        _cs_params = self.COORDINATE_SYSTEM_PARAMETERS

        # Define the coordinate system attribute
        _cs = self.COORDINATE_SYSTEM(**_cs_params)

        # Return the coordinate system. THIS MUST BE MAINTAINED
        # IN SUBCLASSES.
        return _cs

    @abstractmethod
    def _initialize_units(self, *args, **kwargs) -> SimpleNamespace:
        """
        Initialize the default units for the initial conditions. This abstract method should
        specify a :py:class:`SimpleNamespace` object which contains the default units for all of the
        relevant physical dimensions.

        - Subclasses only need to specify the units that are relevant to their needs - not a full coordinate system.
        - A ``length`` unit is required in all unit systems.
        - Whether or not the unit system is mutable to the user is at the discretion of the subclass developer.

        Parameters
        ----------
        args :
            The arguments passed to ``__init__``.
        kwargs :
            The keyword arguments passed to ``__init__``.

        Returns
        -------
        SimpleNamespace
            Units for the initial conditions.

        Notes
        -----
        By default, this method simply initializes the length unit to the default.
        """
        _u = SimpleNamespace(
            length=Unit(self.DEFAULT_LENGTH_UNIT),
            velocity=Unit(self.DEFAULT_VELOCITY_UNIT),
        )
        return _u

    @abstractmethod
    def _subclass_initialize(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        # @@ BASIC SETUP PROCEDURES @@ #
        # Initialization starts with the built-in behaviors that always occur
        # then progresses to additional setup steps. Thus, we start
        # with units and coordinate system configuration.
        #
        # Initialize the coordinate system for this class.
        try:
            self._coordinate_system: "CoordinateSystem" = self._initialize_coordinates(
                *args, **kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize instance of {self.__class__.__name__} due to an error in initialization:\n"
                f"Failed to initialize coordinate system."
            ) from e

        # Initialize the units for this coordinate system.
        try:
            self._units: SimpleNamespace = self._initialize_units(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize instance of {self.__class__.__name__} due to an error in initialization:\n"
                f"Failed to initialize units."
            ) from e

        # @@ MANAGE ARGS / KWARGS @@ #
        # Call down to _subclass_initialize to allow developers to
        # specify a custom version which can initialize various attributes
        # which are relevant to the specific subclass.
        try:
            self._subclass_initialize(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize instance of {self.__class__.__name__} due to an error in initialization:\n"
                f"Failed to perform subclass initialization procedures."
            ) from e

        # @@ CONTAINER INITIALIZATION @@ #
        # Regardless of the IC class, there are always _models and _particles
        # containers in the class. When we add models or link particles, the references
        # are stored here.
        # These should not be memory consuming because the models themselves should only
        # lazy load data.
        self._models: Dict[str, _ModelReference] = {}
        self._particles: Dict[str, ParticleDataset] = {}

    # @@ MODEL MANAGEMENT @@ #
    # All of these methods manage the various models that are connected
    # to the initial conditions.
    @property
    def models(self) -> List[str]:
        """
        A list of the models registered with this initial conditions system.
        """
        return list(self._models.keys())

    def get_model(self, name: str) -> Optional[_ModelReference]:
        """
        Retrieve a model reference by name.

        Parameters
        ----------
        name: str
            The name of the model to retrieve. If the ``name`` specified doesn't correspond
            to a valid model, then a ``KeyError`` is raised.

        Returns
        -------
        _ModelReference
            The reference to the linked model. This object has 4 attributes which can be
            accessed or modified:

            - ``model`` is the actual model object that has been linked.
            - ``position`` is the location in the IC coordinates of the model.
            - ``velocity`` is the velocity (in the IC coordinates) of the model.
            - ``orientation`` is the orientation of the model (in the IC coordinates).

        Raises
        ------
        KeyError
            If ``name`` is not a valid model.

        """
        # Verify that the model actually exists.
        if name not in self._models:
            raise KeyError(f"No linked model with name {name}.")

        # Fetch the model.
        return self._models[name]

    def update_model(
        self,
        name: str,
        position: Optional[np.ndarray] = None,
        velocity: Optional[Union[np.ndarray, unyt_array]] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        """
        Update the IC attributes of a linked model.

        Parameters
        ----------
        name: str
            The name of the model to update. If the ``name`` specified doesn't correspond
            to a valid model, then a ``KeyError`` is raised.
        position: :py:class:`np.ndarray`, optional
            The location in the IC coordinates of the model. These must be specified in the coordinate system
            of the initial conditions as an array.

            .. warning::

                Units are not parsed here because different coordinate systems may have different units on the
                position coordinates. The user should ensure that the position specified is correct as an array matching
                to the units and coordinate systems of the ICs.
        velocity: :py:class:`unyt.unyt_array`, optional
            The velocity (in the IC coordinates) of the model. These can carry units (unlike ``position``); however, if units
            are not provided then default velocity units are assumed.
        orientation: :py:class:`np.ndarray`, optional
            The orientation of the model (in the IC coordinates). This should be a vector (in the initial conditions) which
            points in the direction of the native ``z`` axis in the model.

        Raises
        ------
        KeyError
            If ``name`` is not a valid model.

        """
        # Validate the model name and fetch the reference.
        if name not in self._models:
            raise KeyError(f"No linked model with name {name}.")
        _model_reference = self._models[name]

        if position is not None:
            # Check that the position is valid.
            position = np.asarray(position)
            if (len(position) != self._coordinate_system.NDIM) or (position.ndim != 1):
                raise TypeError(
                    f"Position must be a {self._coordinate_system.NDIM}-vector, not {position.shape}."
                )
            _model_reference.position = np.asarray(position)
        if velocity is not None:
            # Coerce the velocity to ensure that it is valid.
            if hasattr(velocity, "units"):
                velocity = velocity.to_value(self._units.velocity)
            velocity = np.asarray(velocity)

            # Check that it has the correct shape behavior.
            if (len(velocity) != self._coordinate_system.NDIM) or (velocity.ndim != 1):
                raise TypeError(
                    f"Velocity must be a {self._coordinate_system.NDIM}-vector, not {velocity.shape}."
                )
            _model_reference.velocity = np.asarray(velocity)
        if orientation is not None:
            # Ensure that it is a valid array.
            orientation = np.asarray(orientation)
            if (len(orientation) != self._coordinate_system.NDIM) or (
                orientation.ndim != 1
            ):
                raise TypeError(
                    f"Orientation must be a {self._coordinate_system.NDIM}-vector, not {orientation.shape}."
                )
            _model_reference.orientation = np.asarray(orientation)

    def remove_model(self, name: str) -> None:
        """
        Remove a linked model from the initial conditions.

        Parameters
        ----------
        name: str
            The name of the model to update. If the ``name`` specified doesn't correspond
            to a valid model, then a ``KeyError`` is raised.
        """
        if name in self._models:
            _ = self._models.pop(name)
        else:
            raise KeyError(f"Model '{name}' not found.")

    def add_model(
        self,
        model: "Model",
        name: str,
        position: np.ndarray,
        velocity: Union[np.ndarray, unyt_array],
        orientation: Optional[np.ndarray] = None,
    ):
        """
        Add a linked model to the initial conditions.

        Parameters
        ----------
        model: :py:class:`Model`
        name: str
            The name of the model to update. If the ``name`` specified corresponds
            to an existing linked model, then a ``KeyError`` is raised.
        position: :py:class:`np.ndarray`, optional
            The location in the IC coordinates of the model. These must be specified in the coordinate system
            of the initial conditions as an array.

            .. warning::

                Units are not parsed here because different coordinate systems may have different units on the
                position coordinates. The user should ensure that the position specified is correct as an array matching
                to the units and coordinate systems of the ICs.
        velocity: :py:class:`unyt.unyt_array`, optional
            The velocity (in the IC coordinates) of the model. These can carry units (unlike ``position``); however, if units
            are not provided then default velocity units are assumed.
        orientation: :py:class:`np.ndarray`, optional
            The orientation of the model (in the IC coordinates). This should be a vector (in the initial conditions) which
            points in the direction of the native ``z`` axis in the model.

        """
        # Verify the veracity of the model being added. This gets passed down
        # to the abstract method to ensure that things are okay. By default,
        # this does nothing and allows the model simply to pass through.
        self._verify_new_model(model)
        if name in self._models:
            raise KeyError(f"Model '{name}' already exists and is registered.")

        # Construct the position.
        position = np.asarray(position).ravel()
        if len(position) != self._coordinate_system.NDIM:
            raise ValueError(
                f"Position must be a vector of length NDIM, not {len(position)}."
            )

        # Construct the velocity (this should carry no units either)
        if hasattr(velocity, "units"):
            velocity = velocity.to_value(self._units.velocity)
        velocity = np.asarray(velocity).ravel()
        if len(velocity) != self._coordinate_system.NDIM:
            raise ValueError(
                f"Velocity must be a vector of length NDIM, not {len(velocity)}."
            )

        # Manage the orientation specification.
        orientation = np.asarray(orientation).ravel()
        if len(orientation) != self._coordinate_system.NDIM:
            raise ValueError(
                f"Orientation must be a vector of length NDIM, not {len(orientation)}."
            )

        # Construct the model reference object.
        _ = _ModelReference(
            model=model, position=position, velocity=velocity, orientation=orientation
        )

    @abstractmethod
    def _verify_new_model(self, model: "Model"):
        pass

    # @@ PARTICLE MANAGEMENT @@ #
