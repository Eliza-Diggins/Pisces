from pisces.geometry import (
    CartesianCoordinateSystem,
    SphericalCoordinateSystem,
    PolarCoordinateSystem,
    PseudoSphericalCoordinateSystem,
    ProlateSpheroidalCoordinateSystem,
    ProlateHomoeoidalCoordinateSystem,
    OblateHomoeoidalCoordinateSystem,
    OblateSpheroidalCoordinateSystem,
    CartesianCoordinateSystem1D,
    CartesianCoordinateSystem2D,
)
from tests.geometry._base import _TestCoordinateSystem
import numpy as np

class TestCartesianCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = CartesianCoordinateSystem

class TestCartesianCoordinateSystem1D(_TestCoordinateSystem):
    COORD_SYS_CLASS = CartesianCoordinateSystem1D

    def test_gradient(self):
        # GRAB the instance of the coordinate system
        cs = self.coordinate_system

        # CONSTRUCT the function and the array of points.
        f = lambda _x: _x**2
        x = np.linspace(-1, 1, 100)
        y = f(x)

        # CONSTRUCT expectation
        dy = 2*x
        dy = dy.reshape(-1, 1)

        # COMPUTE gradient
        grad_y = cs.gradient(x,y,edge_order=2)

        assert np.allclose(grad_y, dy,rtol=1e-7), f'Gradient for {self.COORD_SYS_CLASS.__name__} did not match theory prediction.'

    def test_divergence(self):
        # GRAB the instance of the coordinate system
        cs = self.coordinate_system

        # CONSTRUCT the function and the array of points.
        f = lambda _x: _x**2
        x = np.linspace(-1, 1, 100)
        y = f(x)
        y = y.reshape(-1, 1)


        # COMPUTE divergence
        div_y = cs.divergence(x,y,edge_order=2)

        assert np.allclose(div_y, 2*x,rtol=1e-7), f'Divergence for {self.COORD_SYS_CLASS.__name__} did not match theory prediction.'

    def test_laplacian(self):
        # GRAB the instance of the coordinate system
        cs = self.coordinate_system

        # CONSTRUCT the function and the array of points.
        f = lambda _x: _x**2
        x = np.linspace(-1, 1, 100)
        y = f(x)

        # CONSTRUCT expectation
        dy = 2*x
        dy = dy.reshape(-1, 1)

        # COMPUTE gradient
        l_y = cs.laplacian(x,y,edge_order=2)

        assert np.allclose(l_y, 2*np.ones_like(l_y),rtol=1e-7), f'Laplacian for {self.COORD_SYS_CLASS.__name__} did not match theory prediction.'

class TestCartesianCoordinateSystem2D(_TestCoordinateSystem):
    COORD_SYS_CLASS = CartesianCoordinateSystem2D

class TestSphericalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = SphericalCoordinateSystem

class TestPolarCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = PolarCoordinateSystem

class TestPseudoSphericalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = PseudoSphericalCoordinateSystem
    PARAMETERS = ([],dict(
        scale_x=2,
        scale_y=1,
        scale_z=3,
    ))

    def test_shell_volume(self):
        instance = self.COORD_SYS_CLASS(scale_x=1,scale_y=1,scale_z=1)
        assert np.isclose(instance._shell_coefficient,4*np.pi), f'{instance._shell_coefficient} != {4*np.pi}'

class TestOblateHomoeoidalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = OblateHomoeoidalCoordinateSystem
    PARAMETERS = ([],dict(ecc=0.95))

    def test_shell_volume(self):
        instance = self.COORD_SYS_CLASS(ecc=0)
        assert np.isclose(instance._shell_coefficient,4*np.pi), f'{instance._shell_coefficient} != {4*np.pi}'

class TestProlateHomoeoidalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = ProlateHomoeoidalCoordinateSystem
    PARAMETERS = ([],dict(ecc=0.95))

    def test_shell_volume(self):
        instance = self.COORD_SYS_CLASS(ecc=0)
        assert np.isclose(instance._shell_coefficient,4*np.pi), f'{instance._shell_coefficient} != {4*np.pi}'

class TestProlateSpheroidalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = ProlateSpheroidalCoordinateSystem
    PARAMETERS = ([],dict(a=1))

class TestOblateSpheroidalCoordinateSystem(_TestCoordinateSystem):
    COORD_SYS_CLASS = OblateSpheroidalCoordinateSystem
    PARAMETERS = ([],dict(a=1))



