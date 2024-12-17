"""
Galaxy cluster utilities.
"""
from pisces.utilities.config import YAMLConfig, config_directory
import os

# @@ Galaxy Cluster Parameters @@ #
# The /bin/galaxy_clusters.yaml file contains the settings for the base
# implementations of galaxy clusters.
gcluster_config_directory = os.path.join(config_directory, 'galaxy_clusters.yaml')
gcluster_params: YAMLConfig = YAMLConfig(gcluster_config_directory)
""" YAMLConfig: The configuration settings for galaxy cluster models."""