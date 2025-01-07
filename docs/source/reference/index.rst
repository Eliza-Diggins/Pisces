.. _reference:
Pisces User Guide
=================

Some areas of the ``Pisces`` code are quite complex and rely heavily on complicated ideas from
astrophysics, mathematics, and numerical methods. To make the use of the code as
straight-forward as possible, we've provided a **User-Guide** for those interested in learning
about particular aspects of the code. Resources in this section range from complex notes about
the Pisces backend to basic physics overviews of specific types of systems.

:material-regular:`rocket_launch;2em` Physical Models And Components
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. raw:: html

   <hr style="height:2px;background-color:black">

At the core of any Pisces model, initial conditions, or other work product is at least one physical model
representing a system or component of a system. This is where the physics really comes into the code and each
of the available models is designed to seamlessly provide for versatile and robust physics to meet your needs.

To get started, go ahead and read the :ref:`modeling_overview` document. Once you've given that a read, the
guides below will introduce the relevant physics and API for specific models.

Backend References
''''''''''''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./models/modeling_grids
    ./models/pipelines_solvers

Overview
''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./models/modeling_overview

Physical Models
'''''''''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./models/galaxy_clusters

.. tip::

    If you're a contributing developer, you'll get a lot out of the developer guide for models: :ref:`modeling_developer`.

:material-regular:`show_chart;2em` Profiles
+++++++++++++++++++++++++++++++++++++++++++

.. raw:: html

   <hr style="height:2px;background-color:black">

Underlying all of the models in Pisces are physical profiles representing individual quantities like density and
temperature. These are all managed in the :py:mod:`~pisces.profiles` module. These guides will cover the basics of working
with these classes.

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./profiles/profiles_overview
    ./profiles/profiles_developer




:material-regular:`functions;2em` Mathematics
++++++++++++++++++++++++++++++++++++++++++++++

.. raw:: html

   <hr style="height:2px;background-color:black">

These documents give an overview of the mathematical details that are relevant for this software.

System Geometry
'''''''''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./geometry/geometry_overview

Mathematical Physics
''''''''''''''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./geometry/poisson_equation

Statistical Methods
'''''''''''''''''''

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./particles/sampling