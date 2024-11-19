from typing import Iterable
from pisces.io.hdf5 import IndexType,  HDF5ElementCache
from pisces.profiles.base import Profile

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

if __name__ == '__main__':
    import h5py
    f = h5py.File('test.hdf5','w')
    f.close()
    f = h5py.File('test.hdf5','r+')

    q = HDF5ProfileRegistry(f)

    from density import NFWDensityProfile

    p = NFWDensityProfile(rho_0=1,r_s=2)

    q['density'] = p
    print(q)

    q.unload_element('density')

    print(q)

    print(q['density'])

    print(q)
