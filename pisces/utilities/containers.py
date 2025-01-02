"""
Generic container classes for use in Pisces.
"""
from collections import OrderedDict
from typing import Any, Optional
from pisces.utilities.general import get_deep_size

class LRUCache(dict):
    """
    A dictionary-based Least Recently Used (LRU) cache with a maximum size.

    This cache stores a limited number of items, automatically removing the least recently used (LRU) items
    when the total size of the cache exceeds the specified maximum size (in MB).

    Parameters
    ----------
    max_size : float, optional
        The maximum size the cache permits, in megabytes (MB). If `max_size` is not specified, the cache
        will have unlimited size and behave like a standard dictionary without automatic eviction.

    Attributes
    ----------
    max_size : float
        The maximum size of the cache in MB.
    _order : collections.OrderedDict
        Tracks the order of access for LRU eviction.

    Methods
    -------
    get(key: Any) -> Optional[Any]
        Retrieve an item from the cache by its key. Marks the item as most recently used.
    put(key: Any, value: Any) -> None
        Add an item to the cache. If the key already exists, it updates the value and marks it as most recently used.
    remove(key: Any) -> None
        Remove an item from the cache by its key.
    clear() -> None
        Remove all items from the cache.
    cache_size -> float
        Return the current size of the cache in MB.
    __len__() -> int
        Return the number of items currently in the cache.
    """

    def __init__(self, max_size: Optional[float] = None):
        if max_size is None:
            max_size = float('inf')  # Unlimited size if not specified

        if max_size <= 0:
            raise ValueError("max_size must be greater than 0.")

        super().__init__()
        self.max_size = max_size
        self._order = OrderedDict()

    def _evict_if_necessary(self):
        """
        Evict items until the cache size is within the maximum size limit.

        Notes
        -----
        - This method uses the `_order` attribute to determine the LRU item.
        - Items are removed in the order of least recent access until the cache size
          is within the specified `max_size`.
        """
        while self.cache_size > self.max_size:
            oldest_key = next(iter(self._order))
            self.__delitem__(oldest_key)

    @property
    def cache_size(self) -> float:
        """
        Compute the current size of the cache in MB.

        Returns
        -------
        float
            The current size of the cache in megabytes.
        """
        return get_deep_size(self) / (1024 * 1024)

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve an item from the cache by its key and mark it as most recently used.

        Parameters
        ----------
        key : Any
            The key of the item to retrieve.

        Returns
        -------
        Any
            The value associated with the key.

        Raises
        ------
        KeyError
            If the key is not found in the cache.
        """
        if key in self._order:
            self._order.move_to_end(key)
            return super().__getitem__(key)
        raise KeyError(f"Key {key} not found in the cache.")

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Add or update an item in the cache.

        Parameters
        ----------
        key : Any
            The key of the item.
        value : Any
            The value of the item.

        Notes
        -----
        - If the key already exists, it updates the value and marks it as most recently used.
        - If the cache size exceeds `max_size`, the least recently used items are evicted.
        """
        if key in self._order:
            self._order.move_to_end(key)
        else:
            self._order[key] = None

        super().__setitem__(key, value)
        self._evict_if_necessary()

    def __delitem__(self, key: Any) -> None:
        """
        Remove an item from the cache by its key.

        Parameters
        ----------
        key : Any
            The key of the item to remove.

        Raises
        ------
        KeyError
            If the key is not found in the cache.
        """
        if key in self._order:
            del self._order[key]
        super().__delitem__(key)

    def clear(self) -> None:
        """
        Remove all items from the cache.

        Notes
        -----
        - This method clears both the internal dictionary and the `_order` tracker.
        """
        self._order.clear()
        super().clear()

    def __len__(self) -> int:
        """
        Get the number of items currently in the cache.

        Returns
        -------
        int
            The number of items in the cache.
        """
        return len(self._order)

class LRUCacheDescriptor:
    """
    Descriptor for managing an LRUCache as an attribute of a class.

    This descriptor ensures that each instance of the owning class gets its
    own LRUCache with the specified maximum size.

    Parameters
    ----------
    max_size : Optional[float]
        The maximum size of the LRUCache in MB. Defaults to no limit.

    Attributes
    ----------
    _max_size : Optional[float]
        The maximum size of the LRUCache in MB.
    _name : str
        The name of the attribute in the owner class.
    _attr_name : str
        The internal name of the attribute where the cache is stored.

    Methods
    -------
    __get__(instance, owner)
        Retrieve the LRUCache for the instance. Creates one if it doesn't exist.
    __set_name__(owner, name)
        Set up the attribute name used for storing the cache internally.
    """

    def __init__(self, max_size: Optional[float] = None):
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be greater than 0.")
        self._max_size = max_size
        self._name = None
        self._attr_name = None

    def __set_name__(self, owner, name):
        """
        Set the name of the attribute being managed by the descriptor.

        Parameters
        ----------
        owner : type
            The owning class where the descriptor is defined.
        name : str
            The name of the attribute in the owner class.
        """
        self._name = name
        self._attr_name = f"_{name}"

    def __get__(self, instance, owner):
        """
        Retrieve the LRUCache for the instance. If the cache doesn't exist, create one.

        Parameters
        ----------
        instance : object
            The instance of the owner class where the descriptor is accessed.
        owner : type
            The owner class of the descriptor.

        Returns
        -------
        LRUCache
            The LRUCache associated with the instance.
        """
        if not hasattr(instance, self._attr_name):
            setattr(instance, self._attr_name, LRUCache(max_size=self._max_size))
        return getattr(instance, self._attr_name)

    def __set__(self, instance, value):
        """
        Prevent direct assignment to the cache attribute.

        Parameters
        ----------
        instance : object
            The instance of the owner class where the descriptor is accessed.
        value : Any
            The value being assigned to the attribute.

        Raises
        ------
        AttributeError
            Always raised to prevent assignment to the cache.
        """
        raise AttributeError(f"Cannot assign to '{self._name}' directly. Use the LRUCache methods.")
