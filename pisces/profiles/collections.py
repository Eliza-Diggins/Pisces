from typing import Iterable, Union, List
from pisces.io.hdf5 import IndexType,  HDF5ElementCache
from pisces.profiles.base import Profile
import numpy as np

class HDF5ProfileRegistry(HDF5ElementCache[str, Profile]):
    def load_element(self, index: str) -> Profile:
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
            profile_info.append([name, profile_type, profile_params or "N/A", str(profile.axes), str(profile.units)])

        # Handle output with or without tabulate
        if not _use_tabulate:
            return profile_info
        else:
            return tabulate(profile_info, headers=["Profile Name", "Type", "Parameters", "Axes", "Units"], tablefmt="grid")
