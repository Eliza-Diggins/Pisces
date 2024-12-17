from pisces.geometry import CoordinateSystem
import h5py

def coordinate_system_answer_testing(
        cs: CoordinateSystem,
        path: str,
        answer_store: bool,
):
    if answer_store:
        with h5py.File(path,'w') as fio:
            cs.to_file(fio,fmt='hdf5')
    else:
        with h5py.File(path,'r') as fio:
            other = CoordinateSystem.from_file(fio,fmt='hdf5')
        assert cs == other, (f'Error during `coordinate_system_answer_testing`: `other` obtained from {path} was not a match'
                             f' for the newly generated coordinate system {cs}.')
