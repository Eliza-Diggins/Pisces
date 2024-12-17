import tempfile

from pisces.geometry import CoordinateSystem
import pytest
import os
from pathlib import Path
from tests.geometry._utils import coordinate_system_answer_testing
import numpy as np

@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class _TestCoordinateSystem:
    # @@ TEST PARAMETERS @@ #
    # These are the parameters that are used to instantiate the
    # unit test class.
    COORD_SYS_CLASS = CoordinateSystem
    PARAMETERS = ([],{})

    # @@ TRANSIENT PROPERTIES @@ #
    _CS_INST = None
    _CS_PATH = None

    # @@ PROPERTIES @@ #
    @property
    def coordinate_system(self):
        return self.__class__._CS_INST

    @property
    def path(self):
        return self._CS_PATH

    # @@ FIXTURES @@ #

    @pytest.fixture(autouse=True)
    def setup_class(self,answer_dir):
        """Setup method to create a coordinate system instance and set file paths."""
        # GENERATE the coordinate system instance.
        if self.__class__._CS_INST is None:
            self.__class__._CS_INST = self.COORD_SYS_CLASS(*self.__class__.PARAMETERS[0],
                                                           **self.__class__.PARAMETERS[1])

        # SET the path.
        self.__class__._CS_PATH = Path(os.path.join(answer_dir,'geometry','coordinate_systems',f'{self.__class__.COORD_SYS_CLASS.__name__}.hdf5'))
        if not self.__class__._CS_PATH.parents[0].exists():
            self.__class__._CS_PATH.parents[0].mkdir(parents=True)

    @pytest.fixture
    def random_coordinates(self):
        _ndim = self.coordinate_system.NDIM
        return np.random.rand(100,_ndim)

    def test_coordinate_system_differences(self,answer_store):
        coordinate_system_answer_testing(self.__class__._CS_INST,self.__class__._CS_PATH,answer_store)

    def test_to_cartesian_and_back(self, random_coordinates, answer_store):
        """Test that converting to Cartesian and back gives consistent results."""
        instance = self._CS_INST
        cartesian_coords = instance.to_cartesian(random_coordinates)
        native_coords = instance.from_cartesian(cartesian_coords)
        np.testing.assert_allclose(native_coords, random_coordinates, atol=1e-6,
                                   err_msg="Conversion to Cartesian and back is inconsistent.")
        # Store results for answer checking
        if answer_store:
            coordinate_system_answer_testing(instance, self._CS_PATH, answer_store)

    def test_lame_coefficients(self, random_coordinates):
        """Test the calculation of Lame coefficients."""
        instance = self._CS_INST
        lame_coeffs = instance.compute_lame_coefficients(random_coordinates)
        assert lame_coeffs.shape == random_coordinates.shape, "Lame coefficients shape mismatch."
        assert np.all(lame_coeffs > 0), "Lame coefficients must be positive."

    def test_jacobian(self, random_coordinates):
        """Test the Jacobian determinant calculation."""
        instance = self._CS_INST
        jacobian = instance.jacobian(random_coordinates)
        assert jacobian.shape == (*random_coordinates.shape[:-1],), "Jacobian shape mismatch."

    def test_surface_element(self, random_coordinates):
        """Test surface element calculation for each axis."""
        instance = self._CS_INST
        for axis in range(instance.NDIM):
            surface_element = instance.surface_element(random_coordinates, axis)
            assert surface_element.shape == (random_coordinates.shape[0],), f"Surface element shape mismatch for axis {axis}."
