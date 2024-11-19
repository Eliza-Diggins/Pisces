from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import unyt
from numpy._typing import NDArray
from pisces.utilities.general import find_in_subclasses
from grids.structs import BoundingBox, DomainDimensions
from pisces.grids.grid_base import GridContainer, Grid
from pisces.io import HDF5_File_Handle

class GridManager:
    """
        Base class for managing Cartesian grids stored in an HDF5 file.

        The `GridManager` class provides a comprehensive interface for managing grids,
        including metadata such as bounding boxes, axes, and grid dimensions. It handles
        both creation and manipulation of grid data stored in an HDF5 file, ensuring consistency
        and extensibility for advanced grid management techniques such as AMR (Adaptive Mesh Refinement).

        Attributes
        ----------
        path : Path
            Path to the HDF5 file managed by this instance.
        handle : HDF5_File_Handle
            Open handle to the HDF5 file.
        GRIDS : GridContainer
            A container for managing individual grids within the file.
        AXES : List[str]
            Names of the axes in the grid.
        BBOX : BoundingBox
            The bounding box of the grid.
        GS : DomainDimensions
            The grid size along each dimension.
        LENGTH_UNIT : unyt.Unit
            The length unit associated with the grid.

        Notes
        -----
        - This class is designed for extensibility, allowing subclasses to implement
          more advanced features such as hierarchical grid structures or custom metadata.
        - It provides utilities for file creation, attribute loading, and grid hierarchy management.

        Examples
        --------
        **Creating a new grid structure**:

        >>> import numpy as np
        >>> import h5py
        >>> path = "test.h5"
        >>> axes = ["x", "y", "z"]
        >>> bbox = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> grid_size = np.array([100, 100, 100])
        >>> manager = GridManager(path, axes=axes, bbox=bbox, grid_size=grid_size, overwrite=True)
        >>> print(manager)
        <GridManager: path=test.h5, NDIM=3>

        **Opening an existing grid structure**:

        >>> manager = GridManager(path)
        >>> print(manager.AXES)
        ['x' 'y' 'z']

        See Also
        --------
        Grid : Represents individual grids within the hierarchy.
        GridContainer : Manages multiple grids within the file.

        """

    def __init__(self,
                 path: Union[str, Path],
                 axes: Optional[List[str]] = None,
                 bbox: Optional[NDArray[np.floating]] = None,
                 grid_size: Optional[NDArray[np.int_]] = None,
                 overwrite: bool = False,
                 **kwargs):
        """
        Initialize the `GridManager`.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file. This path may already exist or will be created otherwise.
            If ``path`` exists and ``overwrite == True``, then the existing file is deleted before loading.
        axes : List[str], optional
            Names of the axes, required for new file creation.

            .. hint::

                The ``axes`` of a :py:class:`GridManager` determine which (cartesian) coordinates are present in
                the grids. This is then used in downstream geometric manipulations when converting to more complex
                coordinate systems.

        bbox : NDArray[np.floating], optional
            Bounding box for the grid, required for new file creation. Should be a
            2D array with shape ``(2, NDIM)``, where the first row specifies the minimum
            coordinates and the second row specifies the maximum coordinates. Pisces will do its best
            to coerce non-numpy types to the correct format.
        grid_size : NDArray[np.int_], optional
            Grid size along each dimension, required for new file creation. Must be
            a 1D array with length equal to the number of dimensions.
        overwrite : bool, default False
            If True, overwrites the existing HDF5 file and creates a new structure.
        kwargs : dict
            Additional metadata attributes (e.g., length unit).

        Raises
        ------
        ValueError
            If the file does not exist and `axes`, `bbox`, or `grid_size` are not provided
            for creating a new file.

        """
        self.path = Path(path)

        # Handle file existence and overwrite behavior
        if self.path.exists() and overwrite:
            self.path.unlink()  # Remove the existing file

        if not self.path.exists():
            # Create a new file and initialize structure
            self.handle = HDF5_File_Handle(self.path, mode='w')
            if axes is None or bbox is None or grid_size is None:
                raise ValueError("`axes`, `bbox`, and `grid_size` must be provided to create a new structure.")
            self.create_empty_structure(self.handle, axes, bbox, grid_size, **kwargs)
            self.handle = self.handle.switch_mode('r+')
        else:
            # Open the existing file for read/write access
            self.handle = HDF5_File_Handle(self.path, mode='r+')

        # @@ LOADING GRID STRUCTURE FROM HDF5 @@ #
        # The methods called in this sequence can be altered in
        # subclasses to provide extensibility to any number of structures.
        # The only RULE for this loading process is that every GridManager must have a
        # GRIDS collection.
        # @@
        # Load the basic attributes.
        self._load_manager_attributes()

        # Load the structure -> results in population of self.GRIDS.
        self._load_grid_hierarchy()

    @classmethod
    def create_empty_structure(cls,
                               handle: HDF5_File_Handle,
                               axes: List[str],
                               bbox: NDArray[np.floating],
                               grid_size: NDArray[np.int_],
                               **kwargs):
        """
        Create an empty grid structure in the HDF5 file.

        This will create the ``BBOX``, ``GS``, ``LUNIT`` and ``AXES`` attributes in ``handle``.

        Parameters
        ----------
        handle : HDF5_File_Handle
            Open handle to the HDF5 file.
        axes : List[str]
            Names of the axes.
        bbox : NDArray[np.floating]
            Bounding box for the grid.
        grid_size : NDArray[np.int_]
            Grid size along each dimension.
        kwargs : dict
            Additional metadata attributes (e.g., length unit).

        """
        # Coerce inputs to consistent types
        bbox = BoundingBox(bbox)
        grid_size = DomainDimensions(grid_size)

        # Write attributes to the file
        handle.attrs['BBOX'] = bbox
        handle.attrs['GS'] = grid_size
        handle.attrs['AXES'] = axes
        handle.attrs['LUNIT'] = str(kwargs.get('length_unit', 'kpc'))
        handle.attrs['CLS_NAME'] = cls.__name__

    def _load_manager_attributes(self):
        """
        Load and validate core attributes from the HDF5 file handle.

        Raises
        ------
        ValueError
            If the dimensions implied by `BBOX`, `GS`, or `AXES` are inconsistent.
        """
        self._BBOX = BoundingBox(self.handle.attrs['BBOX'])
        self._AXES = self.handle.attrs['AXES']
        self._GS = DomainDimensions(self.handle.attrs['GS'])
        self._LUNIT = unyt.Unit(self.handle.attrs['LUNIT'])

        # Validate dimensions and metadata consistency
        self._NDIM = self._GS.size
        if self._BBOX.shape[-1] != self._NDIM:
            raise ValueError(f"Inconsistent dimensions: BBOX implies {self._BBOX.shape[-1]}, "
                             f"but GS implies {self._NDIM}.")
        if len(self._AXES) != self._NDIM:
            raise ValueError(f"Inconsistent dimensions: AXES implies {len(self._AXES)}, "
                             f"but GS implies {self._NDIM}.")


    def _load_grid_hierarchy(self):
        # @@ ALTER IN SUBCLASSES @@
        # The _load_grid_hierarchy method should load the grid structure into the manager. This
        # may be by creating self.GRIDS or by creating self.LEVELS and then grids within each level.
        self.GRIDS = GridContainer(self.handle)

    def __str__(self):
        return f"<GridManager: path={self.path}, NDIM={self._NDIM}>"

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.GRIDS)

    def clear(self):
        """
        Clear all grids in the GridManager.

        Notes
        -----
        This method removes all grids from the HDF5 file but retains the metadata structure.
        """
        for k in self.GRIDS:
            del self.GRIDS[k]

    def sync(self):
        """
        Synchronize the grid container with the HDF5 file.

        Notes
        -----
        This ensures that the in-memory representation of the grids is consistent
        with the HDF5 file.
        """
        self.GRIDS.sync()

    def add_grid(self, index: Union[int, Tuple[int, ...]],
                 bbox: NDArray[np.floating],
                 grid_size: NDArray[np.int_],
                 overwrite: bool = False,
                 **kwargs):
        """
        Add a new grid to the GridManager.

        Parameters
        ----------
        index : Union[int, Tuple[int, ...]]
            The index where the new grid will be added.
        bbox : NDArray[np.floating]
            The bounding box for the new grid.
        grid_size : NDArray[np.int_]
            The size of the grid along each dimension.
        overwrite : bool, default False
            If True, overwrites an existing grid at the specified index.
        kwargs : dict
            Additional parameters to be passed to the grid creation.
        """
        self.GRIDS.add_grid(index=index, bbox=bbox, grid_size=grid_size, overwrite=overwrite, **kwargs)

    def close(self):
        """
        Close the HDF5 file and release resources.

        Notes
        -----
        Automatically called when the `GridManager` instance is deleted.
        """
        if self.handle:
            self.handle.close()

    def list_grids(self) -> List[Union[int, Tuple[int, ...]]]:
        """
        List all grids managed by the GridManager.

        Returns
        -------
        List[Union[int, Tuple[int, ...]]]
            A list of grid indices.
        """
        return list(self.GRIDS.keys())

    def get_grid(self, index: Union[int, Tuple[int, ...]]) -> 'Grid':
        """
        Retrieve a specific grid by its index.

        Parameters
        ----------
        index : Union[int, Tuple[int, ...]]
            The index of the grid to retrieve.

        Returns
        -------
        Grid
            The grid at the specified index.

        Raises
        ------
        KeyError
            If the specified grid does not exist.
        """
        if index not in self.GRIDS:
            raise KeyError(f"Grid at index {index} does not exist.")
        return self.GRIDS[index]

    def remove_grid(self, index: Union[int, Tuple[int, ...]]):
        """
        Remove a grid from the GridManager by its index.

        Parameters
        ----------
        index : Union[int, Tuple[int, ...]]
            The index of the grid to remove.

        Raises
        ------
        KeyError
            If the specified grid does not exist.
        """
        if index not in self.GRIDS:
            raise KeyError(f"Grid at index {index} does not exist.")
        del self.GRIDS[index]

    def commit_changes(self):
        """
        Commit changes to the HDF5 file.

        Notes
        -----
        This method ensures that all changes are written to disk.
        """
        if self.handle:
            self.handle.flush()

    @property
    def AXES(self) -> List[str]:
        """List[str]: Names of the axes of the grid."""
        return self._AXES

    @property
    def LENGTH_UNIT(self) -> unyt.Unit:
        """unyt.Unit: Length unit of the grid."""
        return self._LUNIT

    @property
    def GS(self) -> DomainDimensions:
        """DomainDimensions: Grid size along each dimension."""
        return self._GS

    @property
    def BBOX(self) -> BoundingBox:
        """BoundingBox: Bounding box for the grid."""
        return self._BBOX

    @classmethod
    def load_subclass_from_disk(cls, path: Union[str, Path]) -> 'GridManager':
        """
        Load a specific subclass of `GridManager` from an HDF5 file.

        This method dynamically identifies and initializes the appropriate subclass
        of `GridManager` based on the class name stored in the HDF5 file's metadata.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the HDF5 file containing the saved GridManager.

        Returns
        -------
        GridManager
            An instance of the identified subclass initialized with the file.

        Raises
        ------
        ValueError
            If the specified path does not exist or is not an HDF5 file.
        KeyError
            If the required class name metadata (`CLS_NAME`) is missing in the file.
        ImportError
            If the subclass specified in the file cannot be located in the hierarchy.

        Notes
        -----
        - This method relies on the presence of a `CLS_NAME` attribute in the HDF5 file,
          which specifies the class name of the GridManager subclass to load.
        - The `find_in_subclasses` utility searches for the appropriate subclass in the
          inheritance tree of `GridManager`.

        Examples
        --------
        **Load a saved GridManager subclass:**

        >>> manager = GridManager.load_subclass_from_disk("example.h5")
        >>> print(type(manager))
        <class '__main__.CustomGridManager'>
        """
        path = Path(path)

        # Ensure the path exists and points to a file
        if not path.exists() or not path.is_file():
            raise ValueError(f"Invalid path: {path}. Ensure it exists and is a valid file.")

        # Open the file handle and retrieve the class name
        with HDF5_File_Handle(path, mode='r') as handle:
            if 'CLS_NAME' not in handle.attrs:
                raise KeyError("The HDF5 file is missing the required 'CLS_NAME' attribute.")
            class_name = handle.attrs['CLS_NAME']

        # Locate the subclass using the class name
        if class_name == cls.__name__:
            subclass = cls
        else:
            subclass = find_in_subclasses(GridManager, class_name)

        if subclass is None:
            raise ImportError(f"Could not locate a subclass of GridManager with the name '{class_name}'.")

        # Initialize and return an instance of the subclass
        return subclass(path)

