"""
Registry utilities for Pisces profiles.
"""
from typing import Iterable, Union, List
from pisces.io.hdf5 import IndexType,  HDF5ElementCache
from pisces.profiles.base import Profile
import numpy as np

class HDF5ProfileRegistry(HDF5ElementCache[str, Profile]):
    r"""
    A specialized registry for storing and retrieving :py:class:`~pisces.profiles.base.Profile` objects in an HDF5 file.

    This class extends :py:class:`~pisces.io.hdf5.HDF5ElementCache`, using the profile's
    name (a string) as the index/key and the :py:class:`~pisces.profiles.base.Profile`
    instance as the stored element. It manages reading and writing to the underlying HDF5
    handle via the :py:meth:`Profile.from_hdf5` and :py:meth:`Profile.to_hdf5` methods.

    The HDF5 structure is expected to store each profile in a distinct group/dataset under
    the ``_index_to_key(index)`` path, typically matching the ``index`` name.

    Notes
    -----
    - The class relies on the presence of a valid HDF5 file handle (``self._handle``) provided
      by :py:class:`~pisces.io.hdf5.HDF5ElementCache`.
    - The ``Profile`` class must implement ``.from_hdf5(...)`` and ``.to_hdf5(...)``
      for loading/saving.
    """
    def load_element(self, index: str) -> Profile:
        """
        Load a profile from the HDF5 handle by its name (``index``).

        Parameters
        ----------
        index : str
            The name (key) of the profile to load from the HDF5 file.

        Returns
        -------
        Profile
            An instance of :py:class:`Profile` (or subclass) reconstructed from HDF5.

        Raises
        ------
        KeyError
            If the specified profile index does not exist in the HDF5 handle.
        """
        return Profile.from_hdf5(self._handle,index)

    def _set_element_in_handle(self, index: str, value: Profile):
        value.to_hdf5(self._handle, self._index_to_key(index),overwrite=True)

    def _remove_element_from_handle(self, index: str):
        del self._handle[self._index_to_key(index)]

    def _identify_elements_from_handle(self) -> Iterable[IndexType]:
        elements = []
        for element in self._handle.keys():
            elements.append(element)

        return elements

    def add_profile(self, index: str, profile: Profile, overwrite: bool = False):
        """
        Add a new profile to the registry.

        Parameters
        ----------
        index: str
            The name of the profile.
        profile : Profile
            The Profile instance to add to the registry.
        overwrite : bool, optional
            If True, overwrites an existing profile with the same name. Default is False.

        Raises
        ------
        ValueError
            If a profile with the same name already exists and overwrite is False.
        """
        # Use the profile's class name as the default index if not explicitly set
        if index in self._handle and not overwrite:
            raise ValueError(f"A profile with the name '{index}' already exists. Use overwrite=True to replace it.")

        # Save the profile to the HDF5 file
        self[index] = profile

    def remove_profile(self,index: str):
        """
        Remove a profile from the registry by name.

        Parameters
        ----------
        index : str
            The name of the profile to remove.

        Raises
        ------
        KeyError
            If the profile does not exist.
        """
        del self[index]

    def get_profile_summary(self) -> Union[str, List[List[str]]]:
        """
        Generate a summary of the profiles stored in the model.

        This summary includes:
        - Profile Name
        - Profile Type (class name)
        - Profile Parameters

        Returns
        -------
        Union[List[str], List[List[str]]]
            If ``tabulate`` is installed, returns a formatted table as a string.
            Otherwise, returns a nested list of profile information.

        Notes
        -----
        - The method dynamically extracts the profile's parameters using its ``__dict__``.
        - The summary will show profiles stored in the HDF5 profile registry.
        """
        # Import tabulate safely
        try:
            from tabulate import tabulate
            _use_tabulate = True
        except ImportError:
            _use_tabulate = False
            tabulate = None  # To trick the IDE

        # Construct the profile data
        profile_info = []
        for name, profile in self.items():
            profile_type = profile.__class__.__name__
            profile_params = ", ".join(
                f"{k}={np.format_float_scientific(v,precision=3)}" for k, v in profile.parameters.items() if not k.startswith("_")
            )
            profile_info.append([name, profile_type, profile_params or "N/A", str(profile.AXES), str(profile.units)])

        # Handle output with or without tabulate
        if not _use_tabulate:
            return profile_info
        else:
            return tabulate(profile_info, headers=["Profile Name", "Type", "Parameters", "Axes", "Units"], tablefmt="grid")
