from pathlib import Path
from typing import Union, Optional, List, Tuple, Collection

import h5py
import numpy as np
import unyt
from numpy.typing import NDArray
from pisces.io import HDF5_File_Handle
from pisces.geometry.coordinate_systems import CoordinateSystem
from pisces.grids.structs import BoundingBox, DomainDimensions
class ModelGridManager:
    def __init__(self,
                 path: Union[str, Path],
                 /,
                 coordinate_system: CoordinateSystem = None,
                 bbox: Optional[NDArray[np.floating]] = None,
                 domain_shape: Optional[NDArray[np.int_]] = None,
                 spacing: Optional[List[str]] = None,
                 chunk_size: Optional[NDArray[np.int_]] = None,
                 *,
                 overwrite: bool = False,
                 length_unit: str = "kpc"):
        # Setup the path and manage overwriting issues.
        self.path = Path(path)
        if self.path.exists() and overwrite:
            self.path.unlink()

        if not self.path.exists():
            # We need to create a new structure. That requires that coordinate_system, bbox, and domain_shape
            # are present.
            if any(arg is None for arg in [bbox,domain_shape,coordinate_system]):
                raise ValueError(f"Cannot create new ModelGridManager at {path} because not all of `bbox`, `domain_shape` "
                                 f" and `coordinate_system` were provided.")

            # Create the HDF5 structure
            self.handle = HDF5_File_Handle(self.path, mode='w')
            self.build_skeleton(self.handle, coordinate_system, bbox, domain_shape, spacing=spacing,chunk_size=chunk_size, length_unit=length_unit)
            self.handle = self.handle.switch_mode('r+')
        else:
            # Open an existing file
            self.handle = HDF5_File_Handle(self.path, mode='r+')

        # Load attributes
        self._load_attributes()

    @classmethod
    def build_skeleton(cls,
                       handle: Union[h5py.File,HDF5_File_Handle],
                       coordinate_system: CoordinateSystem,
                       bbox: NDArray[np.floating],
                       domain_shape: NDArray[np.int_],
                       spacing: Optional[List[str]] = None,
                       chunk_size: Optional[NDArray[np.int_]] = None,
                       length_unit: str = 'kpc'):
        # Save the coordinate system to disk at /CSYS
        coordinate_system.to_file(handle.require_group('CSYS'),fmt='hdf5')

        # Manage the bounding box and the domain shape.
        bbox, domain_shape = BoundingBox(bbox),DomainDimensions(domain_shape)

        if (bbox.shape[-1] != coordinate_system.NDIM) or (bbox.shape[-1] != len(domain_shape)):
            raise ValueError(f"Inconsistent dimensions: BBOX implies {bbox.shape[-1]}, DDIM implies {len(domain_shape)} and"
                             f" coordinate system implies {coordinate_system.NDIM}.")

        # Write the bounding box and the domain to disk
        handle.attrs['BBOX'] = bbox
        handle.attrs['DOMAIN_SHAPE'] = domain_shape

        # Managing chunk size
        if chunk_size is None:
            chunk_size = domain_shape
        if isinstance(chunk_size,int):
            chunk_size = DomainDimensions(coordinate_system.NDIM*[chunk_size])
        else:
            chunk_size = DomainDimensions(chunk_size)

        if not np.all(np.mod(domain_shape, chunk_size) == 0):
            raise ValueError("`domain_shape` must be integer multiples of `chunk_size` in all dimensions.")

        handle.attrs['CHUNK_SIZE'] = chunk_size

        # Manage the spacing
        if not isinstance(spacing, Collection):
            spacing = [str(spacing)] * coordinate_system.NDIM

        handle.attrs['SPACING'] = spacing

        # Manage the length units
        handle.attrs['LENGTH_UNIT'] = str(length_unit)


    def _load_attributes(self):
        """
        Load grid attributes from the HDF5 file.
        """
        # Load the coordinate system

        self.bbox = BoundingBox(self.handle.attrs["BBOX"])
        self.domain_shape = DomainDimensions(self.handle.attrs["DOMAIN_SHAPE"])
        self.spacing = self.handle.attrs["SPACING"]
        self.length_unit = unyt.Unit(self.handle.attrs["LENGTH_UNIT"])
        self.coordinate_system = CoordinateSystem.from_file(self.handle['CSYS'],fmt='hdf5')

if __name__ == '__main__':
    from pisces.geometry.coordinate_systems import CartesianCoordinateSystem

    q = CartesianCoordinateSystem()

    p = ModelGridManager('test.hdf5',q,bbox=[[0,1],[0,1],[0,1]],domain_shape=[10,10,10],overwrite=True)
    print(p.coordinate_system)