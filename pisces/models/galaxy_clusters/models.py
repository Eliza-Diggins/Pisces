from pisces.models.base import Model
from pisces.models.galaxy_clusters.solvers import DensityTotalDensitySolver


class ClusterModel(Model):
    # All cluster models are radial models in some sense; however,
    # non-spherical models may be radial in geometries which are not
    # spherical.
    # NOTE: CURRENTLY ONLY SPHERICAL SUPPORTED
    ALLOWED_SYMMETRIES = [{'r'}]
    ALLOWED_COORDINATE_SYSTEMS = ['SphericalCoordinateSystem']
    ALLOWED_GRID_MANAGER_CLASSES = None

    SOLVERS = {
        'dens_tdens': DensityTotalDensitySolver
    }

if __name__ == '__main__':
    from pisces.geometry import SphericalCoordinateSystem, GeometryHandler, Symmetry
    coord_system = SphericalCoordinateSystem()
    s = Symmetry(['r'], coord_system)
    handler = GeometryHandler(SphericalCoordinateSystem(),symmetry=s)
    from pisces.grids.manager_base import GridManager
    manager = GridManager('test.hdf5',axes=['x'],bbox=[0,1],grid_size=[1000],overwrite=True)
    handler.to_hdf5(manager.handle.require_group('GEOMETRY'))

    h = ClusterModel('test.hdf5')
