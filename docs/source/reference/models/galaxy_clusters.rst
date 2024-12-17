.. _galaxy_clusters_models::
Galaxy Cluster Modeling
=======================

Overview
--------

Galaxy clusters are the largest dynamically relaxed systems in the universe containing on the order of :math:`10^{3}-10^{4}` galaxies. Despite their
name, these systems are dominated by gas (the intra-cluster medium, ICM) and dark matter (the host halo) which make up 9-15\% and 85-90\% of the
constituent mass respectively. These systems are dynamically bound by gravitation and supported against collapse by the hot gas of the ICM and the
virialization of stellar and dark matter populations.

In Pisces, these models may be initialized under various paradigms and with different properties:

- **Geometry**: Any radial coordinate system (homoeoidal, spherical, etc.) may be used.
- **Equation of State**: Any equation of state for the ICM may be used.
- **Advanced Physics**: Non-thermal pressure sources like magnetic fields may also be added to the models.

The Intracluster Medium (ICM)
''''''''''''''''''''''''''''''

Assuming the intracluster medium of galaxy clusters can be modeled as an
ideal fluid, the momentum density :math:`\rho{\bf v}` of the
gas obeys the Euler momentum equation (here written in conservative form
and ignoring magnetic fields, viscosity, etc.):

.. math::

    \frac{\partial({\rho_g{\bf v}})}{\partial{t}} + \nabla \cdot (\rho_g{\bf v}{\bf v})
    = -\nabla{P} + \rho_g{\bf g}

where :math:`\rho_g` is the gas density, :math:`{\bf v}` is the gas velocity,
:math:`P` is the gas pressure, and :math:`{\bf g}` is the gravitational
acceleration. The assumption of hydrostatic equilibrium implies that
:math:`{\bf v} = 0` and that all time derivatives are zero, giving:

.. math::

    \nabla{P} = \rho_g{\bf g}

This core relationship forms the underpinnings of all :py:class:`~pisces.models.galaxy_clusters.models.ClusterModel` instances.
Effectively, a specific geometry is chosen and a combination of thermodynamic and gravitational quantities are used to solve for
all of the relevant fields of the model.


Generating Models
-----------------

Based on the mathematics above, there are a variety of ways to produce :py:class:`~pisces.models.galaxy_clusters.models.ClusterModel` objects.
Most of the common approaches that see use in practice are built into Pisces.

+---------------------------------+---------------------------------------------------------------------------------------+------------------------------------------------------------------+
| Method                          |                                 Function                                              | Description                                                      |
+=================================+=======================================================================================+==================================================================+
| From :math:`\rho_g`             | :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_tden`      | Generates the galaxy cluster from the gas and dynamical density  |
| and :math:`\rho_{\mathrm{dyn}}` |                                                                                       | profiles. Computes temperature / grav. field.                    |
+---------------------------------+---------------------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`\rho_g`             | :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel.from_dens_and_temp`      | Generates the galaxy cluster from the gas density and temperature|
| and :math:`T_g`                 |                                                                                       | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+---------------------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`S`                  |  :py:meth:`~pisces.models.galaxy_clusters.models.ClusterModel..from_dens_and_entr`    | Generates the galaxy cluster from the gas density and entropy    |
| and :math:`\rho_{g}`            |                                                                                       | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+---------------------------------------------------------------------------------------+------------------------------------------------------------------+

From Temperature and Density
''''''''''''''''''''''''''''

A common approach is the work forward from a known :math:`\rho_g` and :math:`T` as a function of radius in the specified geometry of interest.
Using the condition of hydrostatic equilibrium, the potential, pressure, dynamical mass, and other necessary fields are automatically computed.

.. note::

    **For Developers**: This method underlies all of the ``"_from_dens_and_temp`` pipelines in the model class.

Thermodynamic Properties
########################

The first step in the Density-Temperature (DT) pipeline is to compute the pressure using the relevant equation of state for the
intra-cluster medium.

.. note::

    Currently, Pisces only permits the ideal gas EOS for this purpose.

.. math::

    P(r) = \frac{\rho_g(r) k_b T(r)}{m_p \eta},

where :math:`\eta` is the mean-molecular mass (generally 0.6 for galaxy clusters). We can further manipulate the EOS
to obtain other fields like the entropy from these initial fields.

The Gravitational Field
#######################

Once the pressure is determined, Euler's Equations can be used for an incompressible fluid, yielding

.. math::

    \frac{-\nabla P(r)}{\rho_g} = \nabla \Phi

.. warning::

    **Mathematical Note**: Cluster models are always radial (ellipsoidal, spherical, etc.) and so both :math:`\rho_g`, :math:`T`,
    and :math:`P` are functions of the *effective radius* (:math:`r`); however,

    .. math::

        \nabla \Phi = \frac{-\nabla P}{\rho_g} = \frac{1}{\lambda_r \rho_g} \partial_r P,

    where :math:`\lambda_r` is the relevant Lame coefficient. This results in **symmetry breaking** and will yield a gravitational
    field which is either 1, 2, or 3 dimensional.

The potential may be calculated via quadrature from the known :math:`\nabla P`; however, this will not break the symmetry:

.. math::

    \begin{aligned}
    \frac{\lambda_r^{-1} \partial_r P \hat{\bf e}_r}{\rho} &= \lambda_r^{-1} \partial_r \Phi \hat{\bf e}_r\\
    \Phi &= \int_r^{\infty} \frac{\partial_r P}{\rho} \; dr.
    \end{aligned}

As such, we perform this quadrature directly to obtain the potential.

Dynamical Quantities
####################

As a final stage of the pipeline, we need to take :math:`\Phi` and use it to compute :math:`\rho` and other dynamical
quantities. Naively,

.. math::

    \rho = \frac{\nabla^2 \Phi}{4\pi G};

however, this approach is difficult to implement productively given that the computation of second derivatives brings with
it considerable complications in terms of numerical errors / round-off errors. Instead, we employ Gauss' law:

.. math::

    \int_{V(r=r_0)} \nabla^2 \Phi dV = \int_{\partial V(r=r_0)} \nabla \Phi \cdot \hat{\bf e}_r dA = 4\pi G M(<r).

The trick here is that the symmetry of :math:`P` ensures that the gradient of the potential also points in the correct direction. Thus

.. math::

    M(<r) = \frac{\partial_r \Phi}{4\pi G} \int_{\partial V(r=r_0)} \frac{1}{\lambda_r} dA.

Simplifying

.. math::

    M(<r) = \frac{\partial_r \Phi}{4\pi G} \int_0^{2 \pi} \; d\phi \int_0^{\pi}\; d\theta  \frac{\lambda_\phi \lambda_\theta}{\lambda_r}.

Different coordinate systems will induce different :math:`\lambda_i` and thus lead to different dynamical mass values.

Once the mass has been obtained, the various mass and density components may be deduced. First, we obtain the dynamical density via differentiation:

.. math::

    \rho_{\rm dyn} = \frac{\partial_r M_{\rm dyn}(r)}{dV_{\rm shell}},

where :math:`dV_{\rm shell}` refers to the infinitesimal volume of a small shell of radius :math:`dr`.

From Temperature and Density
''''''''''''''''''''''''''''