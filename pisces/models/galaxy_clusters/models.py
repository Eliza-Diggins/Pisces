from pisces.models.base import Model
from pisces.models.galaxy_clusters.solvers import DensityTotalDensitySolver
from pisces._grids.static import StaticGridManager

class ClusterModel(Model):
    # All cluster models are radial models in some sense; however,
    # non-spherical models may be radial in geometries which are not
    # spherical.
    # NOTE: CURRENTLY ONLY SPHERICAL SUPPORTED
    ALLOWED_SYMMETRIES = [{'phi','theta'}]
    ALLOWED_COORDINATE_SYSTEMS = ['SphericalCoordinateSystem']
    ALLOWED_GRID_MANAGER_CLASSES = ['StaticGridManager']

    SOLVERS = {
        'dens_tdens': DensityTotalDensitySolver
    }

if __name__ == '__main__':
    from pisces.geometry import SphericalCoordinateSystem, GeometryHandler, Symmetry
    coord_system = SphericalCoordinateSystem()
    s = Symmetry(['phi','theta'], coord_system)
    handler = GeometryHandler(SphericalCoordinateSystem(),symmetry=s)
    from pisces._grids.manager_base import GridManager
    manager = StaticGridManager('test.hdf5',axes=['x'],bbox=[0,5000],grid_size=[1000],overwrite=True)
    handler.to_hdf5(manager.handle.require_group('GEOMETRY'))

    from pisces.profiles import NFWDensityProfile

    q = NFWDensityProfile(rho_0=1e5,r_s=200)
    tq = NFWDensityProfile(rho_0=1e6, r_s=200)

    h = ClusterModel('test.hdf5')
    h.profiles.add_profile('density',q,overwrite=True)
    h.profiles.add_profile('total_density',tq,overwrite=True)
    h.solver = 'dens_tdens'
    h.solver.solve(h.grid_manager.GRIDS[0])

    r = h.grid_manager.GRIDS[0].get_coordinates().ravel()
    p = h.grid_manager.GRIDS[0].FIELDS['temperature'][...]

    import matplotlib.pyplot as plt

    plt.semilogx(r,p)
    plt.show()
