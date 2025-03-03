"""
Astrophysical modeling toolkit in Pisces.

.. tip::

    For information about modeling and how it works in Pisces, please read :ref:`modeling_overview`.

Types of Models
---------------
Pisces features a number of modeling modules, including a variety of astrophysical systems and stellar systems.
The table below includes a list of the available modeling toolkits and their degree of implementation. In many cases,
implementation is incomplete but planned.

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: ../images/models/cluster.png

        Galaxy Clusters
        ^^^^^^^^^^^^^^^^

        Model galaxy clusters including aspherical systems and
        non-thermal pressure sources (magnetic, bulk motions).

        :py:mod:`~pisces.models.galaxy_clusters`

        +++

        |partial-support|

    .. grid-item-card::
        :img-top: ../images/models/galaxy.jpeg

        Galaxies
        ^^^^^^^^

        Model galaxies (elliptical / spiral) with complete
        treatment of the ICM, halo, etc.

        :py:mod:`~pisces.models.galaxies`

        +++

        |no-support|

    .. grid-item-card::
        :img-top: ../images/models/cosmology.jpg

        Cosmological
        ^^^^^^^^^^^^^

        Generate initial conditions for cosmological simulations.

        :py:mod:`~pisces.models.cosmology`

        +++

        |no-support|

    .. grid-item-card::
        :img-top: ../images/models/star.jpg

        Stars
        ^^^^^^^^

        Model stellar interiors

        :py:mod:`~pisces.models.stars`

        +++

        |no-support|


.. |full-support| image:: https://img.shields.io/static/v1?label="Support"&message="Full"&color="green"
.. |partial-support| image:: https://img.shields.io/static/v1?label="Support"&message="Partial"&color="orange"
.. |no-support| image:: https://img.shields.io/static/v1?label="Support"&message="None"&color="black"
"""
from pisces.models import galaxy_clusters
