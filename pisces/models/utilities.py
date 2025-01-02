"""
Utility functions and classes for Pisces models.
"""
from typing import Optional, TYPE_CHECKING
import warnings
from pisces.utilities.config import YAMLConfig, config_directory
from pathlib import Path
import os

if TYPE_CHECKING:
    from pisces.models.base import Model

class ModelConfigurationDescriptor:
    """
    The :py:class:`ModelConfigurationDescriptor` class serves as a descriptor for managing and accessing YAML configuration
    files within Pisces models. It encapsulates the logic for loading configuration data from a specified file,
    ensuring that configurations are loaded lazily and cached for efficient reuse.

    Parameters
    ----------
    filename : str
        The name of the YAML configuration file to be managed.
         This filename is combined with a predefined ``config_directory`` to determine the full path to the configuration file.

    Attributes
    ----------
    filename : str
        Stores the name of the configuration file provided during initialization.

    path : Path
        Represents the full filesystem path to the configuration file. It is constructed by joining the
         ``config_directory`` with the provided ``filename``.

    _reference : Optional[YAMLConfig]
        A private attribute that holds the reference to the loaded YAML configuration.
         It is initialized as `None` and is populated upon the first access to the descriptor.

    Notes
    -----

    - **Configuration Directory**:
      The `config_directory` variable must be defined in the scope where `ModelConfigurationDescriptor` is used.
      It should point to the directory containing all configuration files.
    """
    def __init__(self, filename: Optional[str] = None):
        """
        Initialize the ModelConfigurationDescriptor.

        Parameters
        ----------
        filename : str, optional
            The name of the YAML configuration file to be managed. This filename is combined with a predefined
            ``config_directory`` to determine the full path to the configuration file.

            If no ``filename`` is provided, it is assumed that there is no configuration file for this model
            and therefore any configuration dependent operations will not work.
        """
        # Set the filename object.
        if filename is None:
            self.filename: Optional[str] = None
            self.path = None
        else:
            self.filename = str(filename)
            self.path: Optional[Path] = Path(os.path.join(config_directory, self.filename))

        # setup the reference object.
        self._reference: Optional[YAMLConfig] = None

    def __get__(self, _, owner: 'Model') -> Optional[YAMLConfig]:
        """
        Retrieve the YAML configuration.

        If the configuration has not been loaded yet, it initializes and caches the `YAMLConfig` instance.

        Parameters
        ----------
        owner : type
            The owner class where the descriptor is defined.

        Returns
        -------
        YAMLConfig
            An instance of `YAMLConfig` initialized with the path to the configuration file.
        """
        # Validate
        if (self.filename is None) and owner._IS_ABC:
            warnings.warn("This is a dummy YAML configuration descriptor because no filename was provided.")
            return None
        elif self.filename is None:
            raise RuntimeError("This is a dummy YAML configuration descriptor because no filename was provided.")

        # Check for the reference and create it if necessary.
        if self._reference is None:
            try:
                self._reference = YAMLConfig(self.path)
            except Exception as e:
                raise RuntimeError(f"Failed to load model configuration for class {owner.__name__} at {self.path}.") from e

        # Return the configuration.
        return self._reference


