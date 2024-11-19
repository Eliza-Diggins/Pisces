from typing import Union, Optional, Tuple, Iterable, TYPE_CHECKING

import h5py
import numpy as np
from numpy._typing import NDArray

from grids.structs import BoundingBox, DomainDimensions
from pisces.io import HDF5_File_Handle, HDF5ElementCache
from utilities.array_utils import get_grid_coordinates

if TYPE_CHECKING:
    from grids.manager_base import GridManager


# noinspection PyUnresolvedReferences
class Grid:
    """
        Class representation of a single coordinate grid in a hierarchical grid structure.

        In any grid management scheme (AMR, OCTTREE, etc.), the :py:class:`Grid` class represents an individual
        grid patch in the system. It has a specific boundary box (:py:attr:`Grid.BBOX`), a specific shape (:py:attr:`Grid.SHAPE`),
        and may contain any number of fields representing functions over those grid coordinates.

        Attributes
        ----------
        handle : h5py.Group
            The ``hdf5`` object where this grid is stored on disk.
        BBOX : BoundingBox
            Bounding box for the grid.
        GS : DomainDimensions
            Grid size along each dimension.
        CELL_SIZE : NDArray[np.floating]
            Cell size derived from the bounding box and grid size.
        grid_manager : GridManager, optional
            The manager responsible for this grid. If it is specified at ``__init__``, then it can be accessed,
            otherwise it may be ``None``.

        Notes
        -----
        - A `Grid` is typically managed by a `GridContainer` or a `GridManager`.
        - This class directly interacts with HDF5 data structures for efficient storage.

        Examples
        --------

        **Creating a grid**: In this example, we'll generate a 1D grid with 1000 points from 0 to 1.

        >>> import h5py
        >>> with h5py.File('test.hdf5','w') as fio:
        ...     grid = Grid(fio,key='grid',bbox=[0,1],grid_size=[1000])
        ...     print(grid)
        <Grid: [0.]->[1.]>

        See Also
        --------
        GridManager : Manages multiple grids and their hierarchy.
        GridContainer : A container class for handling multiple grids.
        """

    def __init__(self,
                 container_handle: Union[h5py.File, h5py.Group, HDF5_File_Handle],
                 key: Optional[str] = None,
                 bbox: Optional[NDArray[np.floating]] = None,
                 grid_size: Optional[NDArray[np.int_]] = None,
                 overwrite: bool = False,
                 __grid_manager__: Optional['GridManager'] = None):
        """
        Initialize a `Grid` instance.

        Parameters
        ----------
        container_handle : h5py.File, h5py.Group, or HDF5_File_Handle
            Handle to the HDF5 container holding the grid data. This may be either ``h5py.File`` or ``h5py.Group``
            depending on context / preference. The grid will then be positioned at ``handle/key`` in the HDF5 file.
            Each field in the grid will then appear as a dataset in the handle and the various grid attributes will
            be found in the attributes of the hdf5 object.
        key : str, optional
            Key (group name) identifying the grid in the HDF5 container. If None, the provided handle
            is assumed to be the grid itself.
        bbox : NDArray[np.floating], optional
            Bounding box for the grid, required for new grid creation. This should be a ``2,N`` array containing the
            lower left and upper right corners of the bounding box.
        grid_size : NDArray[np.int_], optional
            Grid size along each dimension, required for new grid creation. This should be a ``N,`` array containing
            the number of cells in each axis.

            .. hint::

                In Pisces, we always use cell centered grids, not lattice grids. Thus, there are ``K`` cells and ``K+1`` edges
                in each dimension.

        overwrite : bool, default False
            If True, overwrites the existing grid structure if it already exists.
        __grid_manager__ : GridManager, optional
            Reference to the `GridManager` managing this grid. Used for validation and property access.
        """
        # Determine if the grid exists and initialize its handle
        if key is None:
            self.handle = container_handle
            _handle_exists = True
        else:
            _handle_exists = key in container_handle

            # Remove existing handle if overwrite is requested
            if _handle_exists and overwrite:
                del container_handle[key]
                _handle_exists = False

            # Create or retrieve the grid handle
            self.handle = container_handle.require_group(key)

        # Manage connection with the grid manager if present.
        # Use the two-step init to validate the property on setting.
        self._grid_manager = None
        if __grid_manager__:
            self.grid_manager = __grid_manager__  # Triggers validation


        # Handle creation of the grid structure
        if not _handle_exists:
            if bbox is None:
                raise ValueError("`bbox` must be provided to create a new grid structure.")
            self.create_empty_structure(self.handle,bbox,grid_size)

        # Load grid attributes
        self._load_grid_attributes()

    @classmethod
    def create_empty_structure(cls,
                               handle: h5py.Group,
                               bbox: Optional[NDArray[np.floating]],
                               grid_size: Optional[NDArray[np.int_]] = None):
        """
        Create a skeleton structure for a grid at a particular location in an
        HDF5 file.

        Parameters
        ----------
        handle : h5py.Group
            The handle in which to create the empty structure.
        bbox : NDArray[np.floating]
            Bounding box for the grid. Must be a 2D array with shape (2, NDIM).
        grid_size : NDArray[np.int_]
            Grid size (number of cells along each dimension).

        Examples
        --------

        Let's create a basic grid structure at ``grid_test`` in ``test.hdf5``.

        >>> import h5py
        >>> with h5py.File('test.hdf5','w') as fio:
        ...     handle = fio.require_group('grid_test')
        ...     Grid.create_empty_structure(handle,[[0,1],[0,1]],[100,100])
        ...     print(handle.attrs.keys())
        <KeysViewHDF5 ['BBOX', 'GS']>

        And so we have successfully added the basic structure to the group.

        Raises
        ------
        ValueError
            If the bounding box or grid size is invalid.
        """
        # Write basic attributes to the group
        handle.attrs['BBOX'] = BoundingBox(bbox)
        handle.attrs['GS'] = DomainDimensions(grid_size)

    def _load_grid_attributes(self):
        """
        Load core attributes for the grid from the HDF5 file.

        Attributes loaded include:
        - Bounding box (`BBOX`)
        - Grid size (`GS`)
        - Cell size (`CELL_SIZE`)

        Raises
        ------
        ValueError
            If required attributes (`BBOX` or `GS`) are missing in the HDF5 handle.
        """
        if 'BBOX' not in self.handle.attrs or 'GS' not in self.handle.attrs:
            raise ValueError("The grid handle does not contain required attributes (BBOX or GS).")
        self._BBOX = BoundingBox(self.handle.attrs['BBOX'])
        self._GS = DomainDimensions(self.handle.attrs['GS'])
        self._CELL_SIZE = (self._BBOX[1,:]-self._BBOX[0,:])/self._GS

        # Validate attributes if a grid manager is present
        if self.grid_manager:
            self._validate_attributes_with_manager()

    def _validate_attributes_with_manager(self):
        """
        Validate the grid's attributes against the associated GridManager.

        Raises
        ------
        ValueError
            If the grid's attributes are inconsistent with the GridManager.
        """
        if not self.grid_manager:
            return

        # Ensure bounding box matches the grid manager's dimensions
        if self.BBOX.shape[-1] != len(self.grid_manager.AXES):
            raise ValueError(
                f"Grid BBOX dimensions {self.BBOX.shape[-1]} do not match GridManager AXES {len(self.grid_manager.AXES)}."
            )

        # Ensure grid size matches the grid manager's expectations
        if self.GS.size != self.grid_manager.GS.size:
            raise ValueError(
                f"Grid GS {self.GS.size} does not match GridManager GS {self.grid_manager.GS.size}."
            )

    def close(self):
        """
        Close the grid handle if necessary.

        Notes
        -----
        This method is automatically called when the object is deleted to ensure proper
        resource cleanup.
        """
        if isinstance(self.handle, h5py.File):
            self.handle.close()

    def __str__(self):
        bbox_start = self.BBOX[0, :].ravel()
        bbox_end = self.BBOX[1, :].ravel()
        return f"<Grid: {bbox_start}->{bbox_end}>"

    def __del__(self):
        self.close()

    @property
    def BBOX(self) -> BoundingBox:
        """
        BoundingBox: Bounding box for the grid.
        """
        return self._BBOX

    @property
    def GS(self) -> DomainDimensions:
        """
        DomainDimensions: Grid size for the grid.
        """
        return self._GS

    @property
    def CELL_SIZE(self) -> NDArray[np.floating]:
        """

        """
        return self._CELL_SIZE

    @property
    def grid_manager(self) -> 'GridManager':
        """
        GridManager: The manager responsible for this grid.
        """
        return self._grid_manager

    @grid_manager.setter
    def grid_manager(self, manager: Optional['GridManager']):
        """
        Validate and set the `GridManager` reference.

        Ensures that the current grid is part of the specified manager's HDF5 file.

        Parameters
        ----------
        manager : grids.manager_base.GridManager
            The manager to set for this grid.

        Raises
        ------
        ValueError
            If the grid's handle is not part of the manager's HDF5 structure.
        """
        if manager is not None:
            if not self._validate_with_manager(manager):
                raise ValueError(f"The grid is not part of the provided GridManager's HDF5 structure.")
        self._grid_manager = manager

    def _validate_with_manager(self, manager: 'GridManager') -> bool:
        """
        Validate that the grid's handle is part of the manager's HDF5 structure.

        Parameters
        ----------
        manager : grids.manager_base.GridManager
            The manager to validate against.

        Returns
        -------
        bool
            True if the grid's handle is valid within the manager's structure, False otherwise.
        """
        if not isinstance(manager, GridManager):
            raise TypeError("The provided manager must be an instance of GridManager.")
        if not self.handle:
            return False
        # Ensure that the file names match and the grid is within the manager's handle
        return self.handle.file == manager.handle.file and self.handle.name in manager.handle

    def get_coordinates(self) -> np.ndarray:
        """
        Get the coordinates of each cell in the grid.

        This method calculates the physical coordinates of each cell in the grid based on
        the bounding box (`BBOX`) and grid size (`GS`).

        Returns
        -------
        np.ndarray
            An array of shape `(*GS,NDIM)` containing the coordinates of each cell.

        Examples
        --------
        .. code-block:: python

            coords = grid.get_coordinates()
            print(coords.shape)  # (10, 10, 2)
        """
        return get_grid_coordinates(
            self.BBOX, self.GS, cell_size=self.CELL_SIZE
        )

class GridContainer(HDF5ElementCache[Tuple[int,...],'Grid']):
    """
    A container for managing multiple grids within an HDF5 structure.

    The `GridContainer` is responsible for organizing and accessing multiple grids,
    each identified by a unique index. It supports adding, copying, and retrieving grids.

    Attributes
    ----------
    GRID_PREFIX : str
        Prefix used for naming grids in the HDF5 structure.

    Methods
    -------
    add_grid
        Add a new grid to the container.
    copy_grid
        Copy an existing grid into the container.

    See Also
    --------
    Grid : Represents individual grids.
    GridManager : Higher-level manager for hierarchical grids.
    """
    # Subclasses may choose to change the GRID_PREFIX to allow
    # for different HDF5 structures.
    GRID_PREFIX = 'GRID'

    def _identify_elements_from_handle(self) -> Iterable[Tuple[int,...]]:
        # All of the elements should be 'GRID_M_N_O'
        elements = []
        for element in self._handle.keys():
            if str(element).startswith(self.GRID_PREFIX):
                elements.append(self._key_to_index(element))

        return elements

    def _set_element_in_handle(self, index: Union[int,Tuple[int,...]], value: Grid):
        """
        Add a Grid to the container, ensuring its handle is part of the container.

        Parameters
        ----------
        index : Union[int,Tuple[int,...]]
            The index at which to add the Grid.
        value : Grid
            The Grid instance to add.

        Raises
        ------
        ValueError
            If the Grid's handle is not part of the container's handle.
        """
        # Check if the Grid's handle is part of this container
        index = self.__class__.ensure_index(index)
        if value.handle.parent != self._handle:
            raise ValueError("The Grid's handle is not part of this container's handle.")

        # Add the Grid's handle to the container
        self._handle[self._index_to_key(index)] = value.handle

    def _remove_element_from_handle(self, index: Tuple[int,...]):
        index = self.__class__.ensure_index(index)
        del self._handle[self._index_to_key(index)]

    def _index_to_key(self, index: Tuple[int,...]) -> str:
        index = self.__class__.ensure_index(index)
        return f"{self.GRID_PREFIX}_{'_'.join(map(str, index))}"

    def _key_to_index(self, key: str) -> Tuple[int,...]:
            return tuple(
                map(int, key.split("_")[1:])
            )

    def load_element(self, index: Tuple[int,...]) -> 'Grid':
        index = self.__class__.ensure_index(index)
        return Grid(self._handle,
                    self._index_to_key(index)
                    )

    def copy_grid(self, index: Union[int,Tuple[int,...]], grid: Grid, overwrite: bool = False):
        """
        Copy the data and attributes of a provided Grid into the container at the specified index.

        Parameters
        ----------
        index : Union[int,Tuple[int,...]]
            The index in the container where the Grid should be copied.
        grid : Grid
            The Grid instance to copy into the container.
        overwrite : bool, optional
            If True, overwrites any existing grid at the specified index. Default is False.

        Raises
        ------
        ValueError
            If overwrite is False and a grid already exists at the index.
        """
        index = self.__class__.ensure_index(index)
        target_key = self._index_to_key(index)

        # Handle existing grid at the index
        if target_key in self._handle:
            if not overwrite:
                raise ValueError(f"A grid already exists at index {index}. Use `overwrite=True` to replace it.")
            # Remove the existing group
            del self._handle[target_key]

        # Use h5py's copy method to copy the entire structure
        self._handle.copy(grid.handle, target_key)


    def add_grid(self,
                 index: Union[int,Tuple[int,...]],
                 bbox: NDArray[np.floating],
                 grid_size: NDArray[np.int_],
                 overwrite: bool = False,
                 **kwargs) -> Grid:
        """
        Add a new grid to the container at the specified index.

        Parameters
        ----------
        index : Union[int,Tuple[int,...]]
            The index in the container where the grid should be added.
        bbox : NDArray[np.floating]
            Bounding box for the new grid.
        grid_size : NDArray[np.int_]
            Grid size along each dimension.
        overwrite : bool, optional
            If True, overwrites any existing grid at the specified index. Default is False.

        Returns
        -------
        Grid
            The newly created Grid instance.

        Raises
        ------
        ValueError
            If a grid already exists at the specified index and overwrite is False.

        Examples
        --------
        .. code-block:: python

            bbox = np.array([[0.0, 0.0], [1.0, 1.0]])
            grid_size = np.array([10, 10])
            container.add_grid((0, 0), bbox=bbox, grid_size=grid_size)
        """
        index = self.__class__.ensure_index(index)
        target_key = self._index_to_key(index)

        # Check if the grid already exists
        if target_key in self._handle:
            if not overwrite:
                raise ValueError(f"A grid already exists at index {index}. Use `overwrite=True` to replace it.")
            # Remove the existing group if overwriting
            del self._handle[target_key]

        # Initialize the new Grid instance
        new_grid = Grid(container_handle=self._handle,
                        key=target_key,
                        bbox=bbox,
                        grid_size=grid_size,
                        overwrite=overwrite,
                        **kwargs)

        # Add the grid to the container
        self.sync()

        return new_grid

    @classmethod
    def ensure_index(cls,index: Union[int,Tuple[int,...]]):
        if isinstance(index,int):
            return (index,)
        else:
            return index