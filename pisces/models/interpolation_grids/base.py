from typing import Union, Optional, List
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from pisces.geometry.base import CoordinateSystem
from pisces.io.hdf5 import HDF5_File_Handle
from pisces.grids.structs import BoundingBox, DomainDimensions

class ModelGridManager:

    def __init__(self,
                 path: Union[str, Path],
                 /,
                 coordinate_system: CoordinateSystem = None,
                 bbox: NDArray[np.floating] = None,
                 grid_size: NDArray[np.int_] = None,
                 scale: List[str] = None,
                 overwrite: bool = False,
                 **kwargs):
        self.path = Path(path)

        # Handle file existence and overwrite behavior
        if self.path.exists() and overwrite:
            self.path.unlink()  # Remove the existing file

        if not self.path.exists():
            # Create a new file and initialize structure
            self.handle = HDF5_File_Handle(self.path, mode='w')

            if (bbox is None) or (coordinate_system is None):
                raise ValueError("`bbox` and `coordinate_system`, and `grid_size` must be provided to create a new structure.")
            self.create_empty_structure(self.handle, coordinate_system, bbox, grid_size,scale=scale, **kwargs)
            self.handle = self.handle.switch_mode('r+')
        else:
            # Open the existing file for read/write access
            self.handle = HDF5_File_Handle(self.path, mode='r+')

        # Load the basic attributes.
        self._load_manager_attributes()

    @classmethod
    def create_empty_structure(cls,
                               handle: HDF5_File_Handle,
                               coordinate_system: CoordinateSystem,
                               bbox: NDArray[np.floating],
                               grid_size: NDArray[np.int_],
                               **kwargs):
        # Coerce inputs to consistent types
        bbox = BoundingBox(bbox)
        grid_size = DomainDimensions(grid_size)

        # Validate that bbox and grid_size are valid.
        if bbox.shape[-1] != coordinate_system.NDIM:
            raise ValueError()

        if len(grid_size) != coordinate_system.NDIM:
            raise ValueError()

        # Grab relevant kwargs if they were provided.
        scale = kwargs.get('scale',None)
        if scale is None:
            scale = 'linear'

        if not isinstance(scale,list):
            scale = [scale]*coordinate_system.NDIM





        # Write attributes to the file
        handle.attrs['BBOX'] = bbox
        handle.attrs['GS'] = grid_size
        handle.attrs['AXES'] = axes
        handle.attrs['LUNIT'] = str(kwargs.get('length_unit', 'kpc'))
        handle.attrs['CLS_NAME'] = cls.__name__
