"""
Virializer subclasses for galaxy cluster models.
"""
import os
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import unyt

from pisces.models.virialize import Virializer
from pisces.particles.sampling.sampling import (
    rejection_sample,
    sample_inverse_cumulative,
)
from pisces.particles.sampling.utils import compute_likelihood_from_density
from pisces.utilities.config import YAMLConfig, config_directory

if TYPE_CHECKING:
    from pisces.models.galaxy_clusters.models import ClusterModel


class GalaxyClusterVirializer(Virializer, ABC):
    """
    Specialized virializer for galaxy cluster models.

    This class loads configuration settings from `galaxy_clusters.yaml` configuration and provides
    a base structure for cluster sampling implementations.
    """

    # Load the configuration file containing the details of the
    # sampling procedures.
    _config = YAMLConfig(os.path.join(config_directory, "galaxy_clusters.yaml"))

    # Set up the class variables.
    _VALID_PARTICLE_TYPES = ["dark_matter", "gas", "stars"]
    DEFAULT_DENSITY_FIELD_LUT = _config["density-field-lut"]
    DEFAULT_FIELD_LUT = _config["field-lut"]

    def _validate_model(self, model: "ClusterModel") -> None:
        super()._validate_model(model)

    def _interpolate_field(self, *args, **kwargs):
        return super()._interpolate_field(*args, **kwargs)


class SphericalClusterVirializer(GalaxyClusterVirializer):
    """
    Virializer for spherical galaxy cluster models.

    Because spherical galaxy clusters feature suitable symmetry, this class utilizes the
    very simple 1-D inverse cumulative sampling method to quickly generate particles.

    To virialize particles, Eddington's Formula is used along with an acceptance rejection
    scheme.
    """

    def _sample_particles(self, species: str, num_particles: int):
        # Obtain the result buffer where the particle positions are
        # placed after the operation.
        result_buffer = self.particles[species]["particle_position_native"]

        # Construct the 1-D likelihood for the particle at a given radius using
        # the provided density field and the Jacobian factor (r^2).
        density_field_name = self.density_lut[species]
        x = self.model.grid_manager.get_coordinates(axes=["r"], scaled=False).ravel()
        likelihood = self.model.FIELDS[density_field_name][...].d.ravel() * x**2

        # Fetch the mass for the density field
        mass_field_name = density_field_name.replace("density", "mass")
        total_mass = self.model.FIELDS[mass_field_name][-1]
        mass_per_particle = total_mass / num_particles

        # Now pass the likelihood and the abscissa into the
        # inversion sampling method.
        result_buffer[:, 0] = sample_inverse_cumulative(
            x, likelihood, num_particles, bounds=self.model.grid_manager.BBOX[:, 0]
        )

        # To construct the other two coordinates, we need to randomly sample for phi and then
        # use an inverse cosine transformation to get the theta coordinate.
        result_buffer[:, 1] = np.arccos(
            np.random.uniform(size=num_particles, low=-1, high=1)
        )
        result_buffer[:, 2] = np.random.uniform(
            size=num_particles, low=-np.pi, high=np.pi
        )
        return mass_per_particle


class SpheroidalClusterVirializer(GalaxyClusterVirializer):
    """
    Virializer for spheroidal galaxy cluster models.

    Because the symmetry is not present in these cases, we need to utilize a more sophisticated method.
    """

    def _sample_particles(self, species: str, num_particles: int):
        # Obtain the result buffer where the particle positions are
        # placed after the operation.
        result_buffer = self.particles[species]["particle_position_native"]

        # Construct the likelihood. In this case, the likelihood may be either 1D or 2D
        # depending on the geometry; however, in all non-spherical cases they should be 2D.
        density_field_name = self.density_lut[species]
        axes, likelihood = compute_likelihood_from_density(
            self.model, density_field_name
        )

        # Determine the total mass from the likelihood.
        _cell_volume = np.prod(
            self.model.grid_manager.CELL_SIZE
        )  # Size of each cell in coordinate units.
        _total_mass = np.sum(likelihood) * _cell_volume
        _volume_units = (
            self.model.grid_manager.length_unit**self.model.coordinate_system.NDIM
        )
        _total_mass_units = (
            unyt.Unit(self.model.FIELDS[density_field_name].units) * _volume_units
        )
        _total_mass = unyt.unyt_quantity(_total_mass, _total_mass_units)
        _mass_per_particle = _total_mass / num_particles

        # Obtain the axes mask and the abscissa for the interpolation.
        axes_mask = self.model.coordinate_system.build_axes_mask(axes)
        axes_log_mask = self.model.grid_manager.is_log_mask
        x = self.model.grid_manager.get_coordinates(axes=axes, scaled=True)

        # Pass off to the rejection sampler. This will ensure that
        # r and theta are set correctly. We then just need to randomly sample
        # for the phi coordinate.
        rejection_sample(
            x,
            likelihood,
            num_particles,
            bounds=self.model.grid_manager.SCALED_BBOX[:, axes_mask].T.ravel(),
            result_buffer=result_buffer.buffer,
        )

        # Correct for any logarithmically scaled values
        result_buffer[:, axes_log_mask] = 10 ** result_buffer[:, axes_log_mask]

        # Sample the phi values.
        result_buffer[:, -1] = np.random.uniform(
            size=num_particles, low=-np.pi, high=np.pi
        )
        return _mass_per_particle
