"""
Basic plotting utilities for Pisces
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def construct_subplot(
    figure: Optional[Figure] = None, axes: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Construct or retrieve a Matplotlib subplot.

    This function ensures a ``Figure`` and ``Axes`` object are available for plotting. If no ``Figure`` or ``Axes``
    is provided, a new one is created using the provided keyword arguments.

    Parameters
    ----------
    figure : Optional[Figure], optional
        An existing Matplotlib ``Figure`` object. If ``None``, a new ``Figure`` is created unless ``axes``
        is provided, in which case the ``figure`` attribute of the ``axes`` is used.
    axes : Optional[Axes], optional
        An existing Matplotlib ``Axes`` object. If ``None``, a new ``Axes`` object is added to the ``Figure``.
    **kwargs : dict, optional
        Additional keyword arguments passed to the ``plt.figure`` function if a new ``Figure`` is created.

    Returns
    -------
    Tuple[Figure, Axes]
        A tuple containing the ``Figure`` and ``Axes`` objects.

    Raises
    ------
    ValueError
        If ``axes`` is provided but its associated ``figure`` is not available.

    Notes
    -----
    - If both ``figure`` and ``axes`` are ``None``, a new ``Figure`` and ``Axes`` are created.
    - If ``axes`` is provided but ``figure`` is ``None``, the ``figure`` is inferred from ``axes.figure``.
    - This function allows flexibility when adding or reusing subplots in Matplotlib.

    """
    # Generate the figure if not provided, pass through kwargs.
    if figure is None:
        # Check if we can take the figure from the axes.
        if axes is not None:
            figure = axes.figure
        else:
            figure = plt.figure(**kwargs)

    # Register axes if not provided. Then proceed.
    if axes is None:
        axes = figure.add_subplot()

    return figure, axes
