from pisces.models.base import Solver
from pisces.models.utils import pipeline, state_checker
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pisces.grids.grid_base import Grid
class DensityTotalDensitySolver(Solver):

    @pipeline(geometry='SphericalCoordinateSystem',symmetry={'r'})
    def spherical_pipeline(self,grid: 'Grid'):
        pass

    @state_checker('geometry')
    def geometry_checker(self):
        return self.model.geometry_handler.coordinate_system.__class__.__name__

    @state_checker('symmetry')
    def symmetry_checker(self):
        return {self.model.geometry_handler.coordinate_system.AXES[k] for k in self.model.geometry_handler.symmetry.symmetry_axes}
