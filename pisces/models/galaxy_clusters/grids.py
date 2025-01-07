"""
Grid manager classes for galaxy cluster models.

This module provides a specialized grid manager for galaxy cluster models
in spherical and similar coordinate systems. It includes methods for correcting
bounding boxes, scaling axes, and managing grid operations in a spherical context.


Notes
-----
The :py:class:`ClusterGridManager` class extends :py:class:`~pisces.models.grids.base.ModelGridManager` to adapt it for spherical geometries,
commonly used in galaxy cluster modeling. It ensures the appropriate construction of
bounding boxes and scaling types for spherical grids.
"""
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from pisces.models.grids.base import ModelGridManager


class ClusterGridManager(ModelGridManager):
    r"""
    A specialized grid manager for spherical grids in galaxy cluster models.

    This class provides utility functions to ensure that the grid's bounding box
    and scaling are correctly adapted to a spherical / pseudo-spherical coordinate system.

    Notes
    -----
    - The bounding box created by `correct_bbox` is designed for the first octant
      of a spherical coordinate system.
    - The scaling adjustments made by `correct_scale` ensure that radial axes can
      use logarithmic scaling while angular axes remain linear.
    """

    ALLOWED_COORDINATE_SYSTEMS = [
        "SphericalCoordinateSystem",
        "OblateHomoeoidalCoordinateSystem",
        "ProlateHomoeoidalCoordinateSystem",
    ]
    DEFAULT_SCALE = ["log", "linear", "linear"]
    DEFAULT_LENGTH_UNIT = "kpc"

    @staticmethod
    def correct_bbox(r_min: float, r_max: float) -> NDArray[np.floating]:
        r"""
        Construct the bounding box for the spherical coordinate system.

        This method creates a bounding box for spherical coordinates, where the
        first axis (r) spans from `r_min` to `r_max`, and the angular axes
        (`theta` and `phi`) are restricted to the first octant.

        Parameters
        ----------
        r_min : float
            Minimum radius of the grid (e.g., starting from the origin).
        r_max : float
            Maximum radius of the grid (e.g., outer radial boundary).

        Returns
        -------
        NDArray[np.floating]
            A 2D numpy array representing the bounding box. The first row
            specifies the minimum values for each axis, and the second row
            specifies the maximum values.

        Examples
        --------
        >>> ClusterGridManager.correct_bbox(0.1, 10.0)
        array([[ 0.1       ,  0.        ,  0.        ],
               [10.        ,  1.57079633,  1.57079633]])

        Notes
        -----
        The bounding box follows the format:
        - `[r_min, 0, 0]` for the minimum values of [r, theta, phi].
        - `[r_max, pi/2, pi/2]` for the maximum values of [r, theta, phi].
        This restricts the grid to the first octant.
        """
        return np.array([[r_min, 0, -np.pi], [r_max, np.pi, np.pi]], dtype="f8")

    @staticmethod
    def correct_scale(scale: Union[List[str], str]) -> List[str]:
        r"""
        Ensure the scale is appropriately formatted for spherical coordinates.

        This method ensures that the radial axis can use logarithmic scaling
        (or any specified scale), while the angular axes (`theta` and `phi`)
        are always set to linear scaling.

        Parameters
        ----------
        scale : Union[List[str], str]
            Scaling type for each axis. If a single string is provided, it is
            assumed to apply only to the radial axis.

        Returns
        -------
        List[str]
            A list of scaling types for the spherical coordinate axes in the order
            [r, theta, phi].

        Examples
        --------
        >>> ClusterGridManager.correct_scale("log")
        ['log', 'linear', 'linear']

        >>> ClusterGridManager.correct_scale(["linear", "linear", "linear"])
        ['linear', 'linear', 'linear']

        Notes
        -----
        - The radial axis (r) can use "log" or "linear" scaling.
        - The angular axes (theta and phi) are always set to "linear" scaling.
        - If a single string is provided for the scale, it applies to the radial axis,
          and the angular axes default to "linear".
        """
        if isinstance(scale, str):
            return [scale, "linear", "linear"]
        return scale
