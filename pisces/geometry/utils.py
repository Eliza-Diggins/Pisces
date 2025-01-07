"""
Utilities for geometry management.
"""
from typing import TYPE_CHECKING

from pisces.geometry.exceptions import ConversionError

if TYPE_CHECKING:
    from pisces.geometry.base import CoordinateSystem


class CoordinateConverter:
    r"""
    A converter class for transforming coordinates between different orthogonal coordinate systems.

    The `CoordinateConverter` class provides a means of converting points between two specified coordinate systems,
    such as converting from spherical to Cartesian coordinates. It validates that the conversion is feasible by
    checking for matching dimensionality and ensuring the systems are not identical.

    This converter leverages specialized conversion functions (`to_<target_coord_system>`) if they exist in the
    `input_coord_system` class; otherwise, it defaults to using Cartesian coordinates as an intermediary step.

    Parameters
    ----------
    input_coord_system : CoordinateSystem
        The coordinate system of the input coordinates. This defines the original basis and geometry for the points.
    output_coord_system : CoordinateSystem
        The coordinate system of the output coordinates. This defines the desired basis and geometry for the points.

    Raises
    ------
    ConversionError
        If the input and output coordinate systems are identical (trivial conversion) or if the dimensionalities
        of the two systems do not match.

    Examples
    --------
    >>> input_system = SphericalCoordinateSystem()
    >>> output_system = CartesianCoordinateSystem()
    >>> converter = CoordinateConverter(input_system, output_system)
    >>> spherical_coords = np.array([[1.0, np.pi / 2, 0.0], [2.0, np.pi / 4, np.pi / 2]])
    >>> cartesian_coords = converter(spherical_coords)
    >>> print(cartesian_coords)
    [[1.0, 0.0, 1.0], [1.0, 1.0, 1.41421356]]

    Attributes
    ----------
    _converter_function : callable
        The function used to perform the conversion. It will be a direct call to a specific conversion method
        if one exists, otherwise, it defaults to Cartesian as an intermediary.
    _str : str
        A string representation of the converter, indicating the transformation from input to output coordinate
        system.

    Notes
    -----
    The `CoordinateConverter` class is designed for orthogonal coordinate systems, leveraging the built-in `to_cartesian`
    and `from_cartesian` methods if no direct conversion function exists. Custom conversion functions, named in
    the format `to_<target_coord_system>`, can be implemented in specific coordinate systems for optimized conversion.
    """

    def __init__(
        self,
        input_coord_system: "CoordinateSystem",
        output_coord_system: "CoordinateSystem",
    ):
        # Validate the converter. Check for same coordinate system and validate number of dimensions.
        if input_coord_system == output_coord_system:
            raise ConversionError(
                f"Conversion from {input_coord_system} to {output_coord_system} is trivial."
            )

        if input_coord_system.NDIM != output_coord_system.NDIM:
            raise ConversionError(
                f"Cannot convert from {input_coord_system} to {output_coord_system} because of a "
                f"dimensionality mismatch: {input_coord_system.NDIM}!={output_coord_system.NDIM}..."
            )

        # Setting attributes.
        self._str = f"<Converter: {input_coord_system} -> {output_coord_system}>"

        # Initialize the converter function.
        if hasattr(input_coord_system, f"to_{output_coord_system.__class__.__name__}"):
            # A specific converter function is available.
            self._converter_function = getattr(
                input_coord_system, f"to_{output_coord_system.__class__.__name__}"
            )
        else:
            # Use Cartesian coordinates as an intermediary step.
            self._converter_function = lambda args: output_coord_system.from_cartesian(
                input_coord_system.to_cartesian(args)
            )

    def __call__(self, coordinates):
        r"""
        Convert coordinates from the input coordinate system to the output coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            An array of coordinates in the input coordinate system format.

        Returns
        -------
        NDArray
            The coordinates transformed to the output coordinate system.

        Notes
        -----
        The conversion is applied by calling the `_converter_function` attribute, which may be either a specialized
        conversion function or a default Cartesian-based intermediary transformation.
        """
        return self._converter_function(coordinates)

    def __repr__(self):
        r"""
        String representation of the `CoordinateConverter` instance.

        Returns
        -------
        str
            A string indicating the transformation direction from the input to the output coordinate system.
        """
        return self._str
