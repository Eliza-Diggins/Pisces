.. _galaxy_clusters_models:
Galaxy Cluster Modeling
=======================

Overview
--------

Galaxy clusters are the largest dynamically relaxed systems in the universe containing on the order of
:math:`10^{3}-10^{4}` galaxies. Despite their name, these systems are dominated by gas (the intra-cluster medium, ICM)
and dark matter (the host halo) which make up 9-15\% and 85-90\% of the constituent mass respectively. These systems are
dynamically bound by gravitation and supported against collapse by the hot gas of the ICM and the virialization of
stellar and dark matter populations.

.. note::

    In Pisces, various physical assumptions / paradigms may be used to construct different galaxy clusters with different
    properties. We currently support the following; however, we are working to expand:

    - **Geometry**: Galaxy clusters are currently supported in all of our
      :py:class:`~pisces.geometry.coordinate_systems.PseudoSphericalCoordinateSystem` subclass and our
      spherical coordinate system: :py:class:`~pisces.geometry.coordinate_systems.SphericalCoordinateSystem`.

Physical Properties
-------------------

Galaxy clusters are (in the simplest sense) dominated by two forces: gravity and thermal pressure. Because the
system must be stabilized against collapse, these two forces must be commensurate and, therefore, knowledge of
one can usually inform the other with considerable accuracy. In this section, we will describe the relevant physics of
the gasseous and gravitationally bound components.


The Intracluster Medium (ICM)
''''''''''''''''''''''''''''''

Assuming the intracluster medium of galaxy clusters can be modeled as an
ideal fluid, the momentum density :math:`\rho{\bf v}` of the
gas obeys the Euler momentum equation (here written in conservative form
and ignoring magnetic fields, viscosity, etc.):

.. math::

    \frac{\partial({\rho_g{\bf v}})}{\partial{t}} + \nabla \cdot (\rho_g{\bf v}{\bf v})
    = \nabla{P} + \rho_g{\bf g}

where :math:`\rho_g` is the gas density, :math:`{\bf v}` is the gas velocity,
:math:`P` is the gas pressure, and :math:`{\bf g}` is the gravitational
acceleration. The assumption of hydrostatic equilibrium implies that
:math:`{\bf v} = 0` and that all time derivatives are zero, giving:

.. math::

    \nabla{P} = -\rho_g{\bf g} = \rho_g \nabla \Phi

This core relationship forms the underpinnings of all :py:class:`~pisces.models.galaxy_clusters.models.ClusterModel` instances.
Effectively, a specific geometry is chosen and a combination of thermodynamic and gravitational quantities are used to solve for
all of the relevant fields of the model.

.. raw:: html

   <hr style="color:black">

Generating Models
-----------------

Using the mathematics of hydrostatic equilibrium, it is possible to start with either a set of thermodynamic
properties or a set of both thermodynamic and gravitational properties and solve for all of the relevant
information in the clusters. This is the underlying principle for generating models in Pisces. The model class is the
:py:class:`~pisces.models.galaxy_clusters.models.ClusterModel` class which implements 2 methods for initialization:

- :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_tden`: Utilizes the properties
  :math:`\rho_g` and :math:`\rho_{\rm dyn}` to construct the cluster.
- :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_temp`: Utilizes the :math:`\rho_g` and
  :math:`T_g` to construct the cluster.


From Density and Temperature
''''''''''''''''''''''''''''

Computing the properties of a galaxy cluster from density and temperature represents the most observationally
relevant methodology, connecting the directly observable properties of clusters to the unobservable dark-sector
properties of the gravitationally bound components. In Pisces, there are 4 major steps to this pipeline, all of which
are preformed automatically in the :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_temp` workflow.

1. The **Equation of State** is used to convert the thermodynamic properties (:math:`\rho_g` and :math:`T_g`) into
   the pressure field :math:`P`.
2. Using the pressure from the previous step, the **hydrostatic equilibrium assumption** is applied to compute the
   gravitational potential (:math:`\Phi`) and gravitational field (:math:`\nabla \Phi`) from the pressure field.
3. From the gravitational field and potential, the **Poisson Equation** is solved to determine the relevant dynamical density
   field.
4. Finally, **mass accounting** is used to compute :math:`\rho_{\rm dm}` using our knowledge of the other relevant
   density components.

The following diagram gives a basic overview of this pipeline:

.. image:: ../../diagrams/gclstr_dens_temp_general.svg

.. note::

    The details of the computations vary based on different assumptions about the EOS, the hydrostatic
    condition, and the gravitational theory. Most importantly, many numerical aspects vary based on the
    coordinate system selected and the relevant symmetries that can be applied.

From Density and Total Density
''''''''''''''''''''''''''''''

While construction from :math:`\rho_g` and :math:`T_g` is highly relevant observationally, many cosmological
applications are more attuned to the use of the total density :math:`\rho_{\rm dyn}` and the gas density :math:`\rho_{g}`.
In simple geometries, this approach is also numerically more robust. Like the temperature / density pipeline, the same
4 steps are applied here; however, they appear in a different order:

1. From the dynamical density (:math:`\rho_{\rm dyn}`), the gravitational field and gravitational potential are
   obtained from **Poisson's Equation**.
2. **Mass accounting** is used to compute the dark matter density (:math:`\rho_{\rm dm}`) from the other relevant
   density components. These profiles are then integrated to determine the total mass components.
1. **Hydrostatic equilibrium** is applied to convert the gravitational field into the pressure field.
2. The **Equation of State** is then solved to obtain the temperature.

The following diagram gives a basic overview of this pipeline:

.. image:: ../../diagrams/gclstr_dens_tden_general.svg

A cluster may be generated in this pipeline using the :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_tden` workflow.
