.. _poisson_equation:

Solutions to the Poisson Equation
=================================

One of the most critical problems in gravitational dynamics comes down to solving Poisson's equation, which relates
the gravitational potential :math:`\Phi` to the total density :math:`\rho` of a system:

.. math::

    \nabla^2 \Phi = 4\pi G \rho.

Solving this equation is a deep and nuanced topic with a variety of methods tailored to specific applications. In this
document, we summarize methods relevant to Pisces models and those implemented in this code base. A solid understanding
of the mathematical principles behind these models is essential to ensure Pisces is used correctly and provides the desired
outcomes.

.. note::

    For detailed treatments, see:

    - Binney, J., & Tremaine, S. (2008). "Galactic Dynamics" (2nd ed.). Princeton University Press.
    - Bovy, J. "Galactic Dynamics" (online resource, accessible at `http://galacticdynamics.org <http://galacticdynamics.org>`_).

Poisson's Equation in Radially Symmetric Systems
------------------------------------------------

One of the most emblematic cases of solving Poisson's equation arises in systems with radial symmetry. In these cases, we
frequently encounter formulations of the Poisson problem which are tractable using 1D quadrature or even analytic techniques.
Because of this, Pisces exploits these scenarios as frequently as possible to avoid the much more complex and expensive procedures
for solving the Poisson problem in very general cases.

.. note::

    Radial symmetry may refer to spherical symmetry or more complex ellipsoidal symmetries such as
    :py:class:`~pisces.geometry.coordinate_systems.OblateHomoeoidalCoordinateSystem`. In Pisces, the :py:class:`~pisces.geometry.base.RadialCoordinateSystem`
    class provides the requisite structure for such systems.


Spherically Symmetric Case
''''''''''''''''''''''''''

The simplest case of Poisson's equation is for systems with spherical symmetry. Two critical theorems attributed to Newton apply:

1. **Newton's 1st Theorem**: A body inside a spherical shell of matter experiences no gravitational force from the shell.
2. **Newton's 2nd Theorem**: A body outside a closed spherical shell experiences the same gravitational acceleration as if the shell's mass were concentrated at its center.

From these theorems, it follows that the potential inside a spherical shell is constant.

To derive the gravitational potential for a density distribution :math:`\rho(r)`, we divide the distribution into infinitesimal spherical shells of width :math:`dr`. Consider a shell with radius :math:`r'` and thickness :math:`dr'`. The mass of this shell is:

.. math::

    dm = 4 \pi r'^2 \rho(r') dr'.

For a point at a distance :math:`r` from the center, the gravitational potential contribution from this shell depends on whether :math:`r < r'` or :math:`r \geq r'`:

.. math::

    \Phi(r, r') = \begin{cases}
    \frac{-G dm}{r}, & r' > r, \\
    \frac{-G dm}{r'}, & r' \leq r.
    \end{cases}

Substituting :math:`dm`:

.. math::

    \Phi(r, r') = \begin{cases}
    \frac{-4\pi G \rho(r') r'^2 dr'}{r}, & r' > r, \\
    -4\pi G \rho(r') r' dr', & r' \leq r.
    \end{cases}

The total gravitational potential :math:`\Phi(r)` is obtained by integrating over all shells, splitting the integral into two parts: :math:`r' \leq r` and :math:`r' > r`:

.. math::

    \Phi(r) = \int_0^r \Phi(r, r') dr' + \int_r^\infty \Phi(r, r') dr'.

Explicitly, this becomes:

.. math::

    \Phi(r) = -4\pi G \left[ \frac{1}{r} \int_0^r r'^2 \rho(r') dr' + \int_r^\infty r' \rho(r') dr' \right].

The first term represents the potential due to the mass enclosed within radius :math:`r`, and the second term accounts for the contribution from the outer shells. This derivation is a direct consequence of the linearity of Poisson's equation and Newton's theorems.

.. note::

    The continuity of :math:`\Phi(r)` ensures a smooth transition between the inner and outer regions, maintaining physical consistency.

For a deeper discussion and numerical examples, refer to Section 2.3 in Binney & Tremaine (2008).


Ellipsoidal Case
''''''''''''''''

A more complex scenario arises in spheroidal or ellipsoidal coordinate systems, such as those described by:

- :py:class:`~pisces.geometry.coordinate_systems.OblateHomoeoidalCoordinateSystem`
- :py:class:`~pisces.geometry.coordinate_systems.ProlateHomoeoidalCoordinateSystem`

Here, extensions of the spherical case arguments are possible but lead to increased mathematical complexity.
For concentric ellipsoids following a density profile :math:`\rho(r)`, where

.. math::

    r^2 = \sum_i \eta_i^2 x_i^2,

the gravitational potential takes the form:

.. math::

    \Phi(\mathbf{x}) = \left(\frac{-\pi G}{\prod_i \eta_i}\right) \int_0^\infty \frac{\psi(\infty) - \psi(\xi(\tau))}{\sqrt{\prod_i \left(\tau + \frac{1}{\eta_i^2}\right)}} d\tau.

Here:

.. math::

    \psi(m) = 2 \int_0^m m \rho(m) dm,

and

.. math::

    \xi^2(\tau) = \prod_i \frac{x_i^2}{\tau + \frac{1}{\eta_i^2}}.

For a more detailed discussion of the mathematical underpinnings, consult Chapter 2 of Binney & Tremaine (2008).


References
----------

.. [BiTr08] Binney, J., & Tremaine, S. (2008). "Galactic Dynamics" (2nd ed.). Princeton University Press.

.. [Bovy23] Bovy, J. "Galactic Dynamics" (online resource). Retrieved from `http://galacticdynamics.org <http://galacticdynamics.org>`_.
