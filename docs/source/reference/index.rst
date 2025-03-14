.. _reference:
Pisces User Guide
=================

Pisces is a very large and diverse code base and has many interesting capabilities - we could not hope
to cover all of these details in a single document. As such, this :ref:`reference` page has been constructed for
users of Pisces to reference when they are in need of information about particular aspects of Pisces.

If you're just getting started, we suggest checking out the :ref:`getting_started` page before jumping in here;
these guides are tailored to users who already have basic understanding of the Pisces code. We also have a collection
of :ref:`examples` which demonstrate many of the things discussed in the documents below.

Astrophysical Models
--------------------

At the core of Pisces is the concept of a **model**, which represents a self-consistent physical representation of
a system. Pisces provides a lot of models which are built-in as well as a lot of tooling for developing your own models.
The documents in this section cover the physics of each of the built-in modeling modules as well as the modeling backend
and information for those who need to build their own models. For a general overview of modeling in pisces, check out
this introductory document: :ref:`modeling_overview`.

Built-In Models
+++++++++++++++

(*We'll add new sections here as new capabilities are developed*)

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./models/galaxy_clusters

Building Models
+++++++++++++++

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./models/modeling_developer
    ./models/pipelines_solvers
    ./models/modeling_grids


Doing Science With Models
-------------------------

Pisces is science-focused at its core and is developed around the idea of providing a central toolkit for performing
a wide array of analyses / scientific workflows for its models. Once a model has been created, there are many things that
can be done with it. This section of the user guide provides documentation on many of the common use-cases for Pisces models.

In many cases, Pisces is interoperable with external / 3rd party software which provides much of its utility for
scientific projects. In cases where a 3rd party software is used, we encourage the user to become familiar with that software
as well in order to ensure that everything functions as expected.

Converting Models to Particles
++++++++++++++++++++++++++++++

For many workflows, the most versatile option for analysis is to convert Pisces models to particle data that
can then be exported to resources like `yt <https://yt-project.org/>'_ or other analysis packages. Pisces provides support
for particle conversions as well as some more advanced machinery.

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./particles/particles
    ./particles/sampling
    ./particles/virialization

Hydrodynamics Simulations
+++++++++++++++++++++++++

Pisces originally started out as an initial-conditions generator for hydrodynamics / MHD simulations. One of the most
useful functionalities of Pisces is its ability to produce these initial conditions and to interface with a variety
of simulation codes.

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./simulations/codes
    ./simulations/initial_conditions

Building Blocks
-------------------------

Behind the scenes, there is a lot going on in Pisces. Users need to be able to generate profiles, handle units, work
in different geometries, etc. all with the end goal of building models they can use for science. In this section, we'll cover
the details of the various "building-blocks" that make up the backend of Pisces.

Profiles
++++++++

Profiles are a critical ingredient necessary for building most if the Pisces models.

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./profiles/profiles_overview
    ./profiles/profiles_developer

Coordinate Systems
++++++++++++++++++

Coordinate system support is one of the most sophisticated aspects of the Pisces code-base but is also one of the
keystones of the entire system. In this section, we'll cover the details of these workhorse classes as well as how to
develop custom ones.

.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./geometry/geometry_overview
    ./geometry/geometry_background


Miscellaneous Guides
--------------------

There are a few guides that don't fit in elsewhere!


.. toctree::
    :titlesonly:
    :glob:
    :numbered:

    ./geometry/poisson_equation
