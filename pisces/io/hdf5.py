import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generic, Iterable, TypeVar, Union

import h5py


class HDF5_File_Handle(h5py.File):
    r"""
    A handler for managing datasets and groups within an HDF5 file or group.
    Supports dynamic creation, resizing, and handling of temporary data storage.

    This class inherits from `h5py.File`, so all methods of `h5py.File` are available.
    Additionally, it includes methods for file deletion and mode switching, and it
    ensures that files are properly closed when the object is deleted.
    """

    def __del__(self):
        """
        Ensure the HDF5 file is closed properly when the object is deleted.

        This is automatically called by Python's garbage collector when the object
        is no longer in use, ensuring that the file handle is released.
        """
        self.close()

    def __str__(self) -> str:
        """
        Return a string representation of the HDF5_File_Handle instance.

        Returns
        -------
        str
            A simplified string representation, showing the file path.
        """
        return f"<HDF5_File_Handle '{self.filename}'>"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the HDF5_File_Handle instance.

        Returns
        -------
        str
            A string representation using `__str__`, primarily for debugging.
        """
        return self.__str__()

    def delete_file(self):
        """
        Completely remove the HDF5 file from disk if it was opened from a file path.

        This method:
        - Closes the file handle.
        - Deletes the file from the filesystem.

        Raises
        ------
        OSError
            If there is an issue removing the file from the filesystem.
        """
        try:
            # Close the file to release the handle.
            self.close()
            # Remove the file from the filesystem.
            os.remove(self.filename)
            print(f"File '{self.filename}' has been deleted from disk.")
        except OSError as e:
            raise OSError(f"Error deleting the file '{self.filename}': {e}")

    def switch_mode(self, mode: str) -> "HDF5_File_Handle":
        """
        Switch the mode of the HDF5 file. This method closes the current file
        handle and reopens the file in the specified mode.

        Parameters
        ----------
        mode : str
            The new mode to reopen the file in. Typical modes include:
            - 'r': Read-only, file must exist.
            - 'r+': Read/write, file must exist.
            - 'w': Create file, truncate if exists.
            - 'w-': Create file, fail if exists.
            - 'a': Read/write if exists, create otherwise.

        Returns
        -------
        HDF5_File_Handle
            A new `HDF5_File_Handle` instance with the specified mode.

        Raises
        ------
        ValueError
            If trying to switch to the same mode as the current one.
        """
        if self.mode == mode:
            return self  # No need to switch if already in the desired mode.

        # Save the filename and reopen in the new mode.
        _filename = self.filename
        self.close()
        return HDF5_File_Handle(_filename, mode=mode)


IndexType = TypeVar("IndexType")
"""
Generic type variable representing the index type of elements in a container.
Used for type hinting in element containers to specify the type of index used for accessing elements.
"""

ItemType = TypeVar("ItemType")
"""
Generic type variable representing the type of elements stored in a container.
Used for type hinting in element containers to specify the type of elements being managed.
"""


class HDF5ElementCache(
    OrderedDict[IndexType, ItemType], Generic[IndexType, ItemType], ABC
):
    """
    Abstract Base Class for an LRU Cache for HDF5 elements.

    This class provides a robust and extensible framework for managing elements stored in HDF5 files or groups.
    It combines the functionality of an `OrderedDict` with an LRU (Least Recently Used) caching mechanism
    to efficiently handle memory usage while allowing subclasses to define rules for interacting with HDF5 data.

    Key Features
    ------------
    - Efficient LRU caching mechanism for memory management.
    - Extensible design with abstract methods for subclass customization.
    - Supports dynamic loading/unloading of elements during iteration.
    - Handles interaction with elements stored in HDF5 structures such as datasets, groups, or other HDF5 objects.

    Parameters
    ----------
    handle : h5py.Group or h5py.File
        Handle to the HDF5 group or file containing the elements to be cached.
    kwargs : dict, optional
        Additional parameters for cache configuration:
        - `cache_size` : int, optional
            Maximum number of elements to keep in memory. Defaults to None (no limit).

    Attributes
    ----------
    _handle : h5py.Group or h5py.File
        The HDF5 handle containing the elements to be cached.
    _CACHE_SIZE : int or None
        Maximum number of elements to cache in memory. If `None`, no cache size limit is enforced.

    Abstract Methods
    ----------------
    - :py:meth:`load_element`: Defines how elements are loaded from the HDF5 handle.
    - :py:meth:`_set_element_in_handle`: Defines how elements are written to the HDF5 handle.
    - :py:meth:`_remove_element_from_handle`: Defines how elements are removed from the HDF5 handle.
    - :py:meth:`_identify_elements_from_handle`: Defines how elements are identified in the HDF5 handle.

    Examples
    --------
    **Subclass Example for HDF5 Datasets**

    .. code-block:: python

        import h5py

        class DatasetCache(HDF5ElementCache[str, h5py.Dataset]):
            def load_element(self, index: str) -> h5py.Dataset:
                return self._handle[index]

            def _set_element_in_handle(self, index: str, value: h5py.Dataset):
                self._handle[index] = value

            def _remove_element_from_handle(self, index: str):
                del self._handle[index]

            def _identify_elements_from_handle(self):
                return [key for key, obj in self._handle.items() if isinstance(obj, h5py.Dataset)]

        # Example usage
        with h5py.File("example.h5", "r") as file:
            cache = DatasetCache(handle=file, cache_size=10)
            dataset = cache["my_dataset"]  # Dynamically loads "my_dataset"
            print(dataset.shape)

    See Also
    --------
    - :py:class:`h5py.File`
    - :py:class:`h5py.Group`
    - :py:class:`OrderedDict`

    Notes
    -----
    - This class is designed to act as a blueprint for caching systems for HDF5-based data structures.
    - Subclasses must implement abstract methods to specify rules for interacting with HDF5 elements.
    """

    def __init__(self, handle: Union[h5py.Group, h5py.File], **kwargs):
        """
        Initialize the HDF5 element cache.

        Parameters
        ----------
        handle : h5py.Group or h5py.File
            Handle to the HDF5 group or file containing elements.
        kwargs : dict
            Additional parameters for cache configuration:
            - `cache_size` : int, optional
                Maximum number of elements to cache in memory. Defaults to None (no limit).

        Examples
        --------
        .. code-block:: python

            with h5py.File("example.h5", "r") as file:
                cache = DatasetCache(handle=file, cache_size=10)
        """
        super().__init__()
        self._handle = handle
        self._CACHE_SIZE = kwargs.get("cache_size", None)

        # Populate the cache keys based on elements in the HDF5 handle
        self._initialize_cache_keys()

    def _initialize_cache_keys(self):
        """
        Populate the cache with references to elements in the HDF5 handle.

        This method identifies elements using `_identify_elements_from_handle` and initializes
        the cache with unloaded elements (`None`).

        Notes
        -----
        - This ensures that all elements in the HDF5 handle are represented in the cache.
        - The elements are not loaded into memory until explicitly accessed.

        Examples
        --------
        .. code-block:: python

            cache = DatasetCache(handle=file)
            print(list(cache.keys()))  # Lists all dataset names in the HDF5 file
        """
        for index in self._identify_elements_from_handle():
            OrderedDict.__setitem__(self, index, None)

    def _key_to_index(self, key: str) -> IndexType:
        """
        Convert a key from the HDF5 handle to an index for the cache.

        Parameters
        ----------
        key : str
            The key in the HDF5 handle.

        Returns
        -------
        IndexType
            The index used for the cache.

        Notes
        -----
        - By default, this method directly returns the key, but subclasses may override it
          to apply custom transformations.
        """
        return key

    def _index_to_key(self, index: IndexType) -> str:
        """
        Convert a cache index back to a key for the HDF5 handle.

        Parameters
        ----------
        index : IndexType
            The index used for the cache.

        Returns
        -------
        str
            The key in the HDF5 handle.

        Notes
        -----
        - By default, this method directly returns the index, but subclasses may override it
          to apply custom transformations.
        """
        return index

    @abstractmethod
    def load_element(self, index: IndexType) -> ItemType:
        """
        Load an element from the HDF5 handle into memory.

        Parameters
        ----------
        index : IndexType
            The index of the element to load.

        Returns
        -------
        ItemType
            The loaded element.

        Notes
        -----
        - Subclasses must implement this method to define how elements are loaded from the HDF5 structure.
        """
        pass

    def unload_element(self, index: IndexType):
        """
        Unload an element from memory.

        This method replaces the element in the cache with `None`, preserving the key
        while releasing the memory associated with the element.

        Parameters
        ----------
        index : IndexType
            The index of the element to unload.

        Notes
        -----
        - Subclasses should ensure that the element is properly unloaded, and any
          necessary HDF5 references remain intact.
        - Unloading does not remove the element from the HDF5 file; it only releases
          memory in the cache.

        """
        if index in self:
            super().__setitem__(index, None)
        else:
            raise KeyError(f"Element with index '{index}' does not exist in the cache.")

    @abstractmethod
    def _set_element_in_handle(self, index: IndexType, value: ItemType):
        """
        Write an element to the HDF5 handle.

        Parameters
        ----------
        index : IndexType
            The index of the element to write.
        value : ItemType
            The element to write to the HDF5 handle.
        """
        pass

    @abstractmethod
    def _remove_element_from_handle(self, index: IndexType):
        """
        Remove an element from the HDF5 handle.

        Parameters
        ----------
        index : IndexType
            The index of the element to remove.
        """
        pass

    @abstractmethod
    def _identify_elements_from_handle(self) -> Iterable[IndexType]:
        """
        Identify elements in the HDF5 handle.

        Returns
        -------
        Iterable[IndexType]
            An iterable of keys representing elements in the HDF5 handle.

        Notes
        -----
        - Subclasses must implement this method to specify how elements are identified in the HDF5 handle.
        """
        pass

    def __getitem__(self, index: IndexType) -> ItemType:
        """
        Retrieve an element, loading it from HDF5 if necessary.

        Parameters
        ----------
        index : IndexType
            The index of the element to retrieve.

        Returns
        -------
        ItemType
            The requested element.

        Raises
        ------
        KeyError
            If the index does not exist in the cache.

        Notes
        -----
        - If the element is not already in memory, it is loaded from the HDF5 handle.
        """
        if index not in self:
            raise KeyError(f"{self} does not contain '{index}'.")
        if super().__getitem__(index) is None:
            super().__setitem__(index, self.load_element(index))
            self._update_cache(index)

        return super().__getitem__(index)

    def __setitem__(self, index: IndexType, value: ItemType):
        """
        Set an element in the cache, updating both memory and HDF5.

        Parameters
        ----------
        index : IndexType
            The index of the element to set.
        value : ItemType
            The element to store.

        Notes
        -----
        - Updates both the in-memory cache and the corresponding element in the HDF5 handle.
        """
        super().__setitem__(index, value)
        self._set_element_in_handle(index, value)
        self._update_cache(index)

    def __delitem__(self, index: IndexType):
        """
        Remove an element from the cache and the HDF5 handle.

        Parameters
        ----------
        index : IndexType
            The index of the element to remove.

        Notes
        -----
        - This method removes the element from both the in-memory cache and the HDF5 handle.
        - If the element is not in the cache or the HDF5 handle, a `KeyError` is raised.

        Raises
        ------
        KeyError
            If the element does not exist in the cache or the HDF5 handle.
        """
        # Check if the element exists in the cache
        if index not in self:
            raise KeyError(f"Element with index '{index}' does not exist in the cache.")

        # Remove the element from the cache
        super().__delitem__(index)

        # Remove the corresponding element from the HDF5 handle
        try:
            self._remove_element_from_handle(index)
        except KeyError:
            raise KeyError(
                f"Element with index '{index}' does not exist in the HDF5 handle."
            )

    def _update_cache(self, index: IndexType):
        """
        Update the cache to reflect recent access, enforcing the cache size limit.

        This method ensures that the cache maintains the LRU (Least Recently Used) order by
        moving the recently accessed or added item to the end of the `OrderedDict`. If the
        cache size exceeds the configured `_CACHE_SIZE`, the least recently used element
        is evicted.

        Parameters
        ----------
        index : IndexType
            The index of the element that was accessed or added.

        Notes
        -----
        - When the cache exceeds `_CACHE_SIZE`, the least recently used element is unloaded
          and removed from the cache.
        - Eviction ensures that the cache remains efficient for memory usage.
        """
        if self._CACHE_SIZE is not None and len(self) > self._CACHE_SIZE:
            # Evict the least recently used element
            lru_key, _ = self.popitem(last=False)  # Pop the first element (LRU)
            self.unload_element(lru_key)

        # Move the accessed or added item to the end of the OrderedDict
        self.move_to_end(index)

    def sync(self):
        """
        Synchronize the cache with the HDF5 handle.

        This method identifies any new elements in the HDF5 handle that are not yet
        part of the cache and adds them. It does not remove elements from the cache
        that are no longer in the handle.

        Notes
        -----
        - New elements are initialized with `None` in the cache and loaded on demand.
        - This method ensures the cache stays up to date with the HDF5 structure.

        Examples
        --------
        .. code-block:: python

            cache.sync()
            # Ensures the cache contains all elements in the HDF5 handle.
        """
        # Identify all elements currently in the HDF5 handle
        current_elements = set(self._identify_elements_from_handle())

        # Identify elements already in the cache
        cached_elements = set(index for index in self.keys())

        # Determine new elements that need to be added
        new_elements = current_elements - cached_elements

        # Add new elements to the cache, initialized with `None`
        for index in new_elements:
            super().__setitem__(index, None)

    def values(self, dynamic_loading: bool = True):
        """
        Override the values() method to lazily load elements during iteration.

        Parameters
        ----------
        dynamic_loading : bool, optional
            If True, dynamically load and unload elements. If False, use current load state only.

        Yields
        ------
        ItemType
            The value corresponding to each key, loading the element if it's not already in memory.
        """
        state_cache = {key: (value is not None) for key, value in super().items()}

        for key, was_loaded in state_cache.items():
            yield self[key]  # This triggers loading if needed

            # Unload if it was originally unloaded and dynamic loading is enabled
            if dynamic_loading and not was_loaded:
                self.unload_element(key)

    def items(self, dynamic_loading: bool = True):
        """
        Override the items() method to lazily load elements during iteration.

        Parameters
        ----------
        dynamic_loading : bool, optional
            If True, dynamically load and unload elements. If False, use current load state only.

        Yields
        ------
        Tuple[IndexType, ItemType]
            The key-value pairs where the value is loaded lazily if necessary.
        """
        state_cache = {key: (value is not None) for key, value in super().items()}

        for key, was_loaded in state_cache.items():
            yield key, self[key]  # This triggers loading if needed

            # Unload if it was originally unloaded and dynamic loading is enabled
            if dynamic_loading and not was_loaded:
                self.unload_element(key)

    def keys(self, dynamic_loading: bool = True):
        """
        Override the keys() method. No need for lazy loading here since it's just returning the keys.

        Returns
        -------
        Iterator[IndexType]
            An iterator over the keys of the container.
        """
        state_cache = {key: (value is not None) for key, value in super().items()}

        for key, was_loaded in state_cache.items():
            yield key

            # Unload if it was originally unloaded and dynamic loading is enabled
            if dynamic_loading and not was_loaded:
                self.unload_element(key)

    def get(self, key: IndexType, default: ItemType = None):
        """
        Return the value for ``key`` if ``key`` is in the dictionary, else ``default``.

        Parameters
        ----------
        key: IndexType
        default: ItemType

        Returns
        -------
        ItemType
            The element in the HDF5 structure or ``default``.
        """
        try:
            return self[key]
        except KeyError:
            return default
