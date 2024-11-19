from typing import TYPE_CHECKING, Type


if TYPE_CHECKING:
    from pisces.profiles.base import ProfileMeta, Profile

class ProfileRegistry:
    """
    A registry class for maintaining collections of :py:class:`pisces.profiles.abc.Profile` classes.

    This class provides methods to register, retrieve, and list all registered
    profile classes.

    Attributes
    ----------
    registry : dict
        A dictionary where the keys are profile class names and the values
        are profile class objects.
    """
    def __init__(self):
        """
        Initialize an empty profile registry.
        """
        self.registry = {}

    def register(self, profile_class: "ProfileMeta"):
        """
        Register a profile class in the registry.

        Parameters
        ----------
        profile_class : Type[Profile]
            The profile class to register.

        Raises
        ------
        ValueError
            If a profile class with the same name already exists in the registry.
        """
        profile_name = profile_class.__name__
        if profile_name in self.registry:
            raise ValueError(f"Profile class '{profile_name}' is already registered.")
        self.registry[profile_name] = profile_class

    def get(self, profile_name: str) -> Type["Profile"]:
        """
        Retrieve a profile class by its name.

        Parameters
        ----------
        profile_name : str
            The name of the profile class to retrieve.

        Returns
        -------
        Type[Profile]
            The profile class corresponding to the provided name.

        Raises
        ------
        KeyError
            If the profile class is not found in the registry.
        """
        if profile_name not in self.registry:
            raise KeyError(f"Profile class '{profile_name}' is not registered.")
        return self.registry[profile_name]

    def list_profiles(self) -> list:
        """
        List all registered profile class names.

        Returns
        -------
        list
            A list of all registered profile class names.
        """
        return list(self.registry.keys())

_DEFAULT_PROFILE_REGISTRY = ProfileRegistry()


