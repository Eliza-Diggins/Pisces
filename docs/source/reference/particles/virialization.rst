.. _virialization:

=========================
Virializing Particles
=========================

In most of the cases where Pisces models are useful, we are interested in systems which are dynamically stable. That means
that, when evolved forward in time, the systems are invariant. For collisionless components (like dark matter or stars), this
is an exercise in statistical mechanics and requires considerable effort to ensure. In this document, we'll provide a brief
overview of the relevant theory and a discussion of the techniques employed in Pisces to virialize particle ensembles.

Theory
------

The Collisionless Boltzmann Equation
''''''''''''''''''''''''''''''''''''

The theory of virialization is best described in **phase-space**, let :math:`f({\bf q},{\bf p})` be the number density
of the ensemble of particles. Particles are neither created nor destroyed, so there should be a **conservation law**
constraining the flow of :math:`f`. This equation takes the form

.. math::

    \frac{\partial f}{\partial t} + \dot{\bf q} \frac{\partial f}{\partial {\bf q}} + \dot{\bf p} \frac{\partial f}{\partial {\bf p}} = 0.

This equation is called the **Collisionless Boltzmann Equation (CBE)**. It is the fundamental equation relevant to
the dynamics of collisionless systems. There are a few things to note about this equation:

1. While it has been written in terms of :math:`{\bf p}` and :math:`{\bf q}`, suggesting canonical coordinates, this is
   actually not a requirement. The CBE holds in any phase space.
2. The CBE is an equation in :math:`2N+1` unknowns, which means that it is (generically) unsolvable without further constraint.

The goal of virialization is to determine :math:`f` sufficiently well to produce a sample of particles from it in equilibrium.

In the traditional phase space :math:`({\bf x},\dot{\bf x})` the CBE takes the very familiar form

.. math::

    \frac{\partial f}{\partial t} + \dot{\bf x} \cdot \frac{\partial f}{\partial {\bf x}} + \dot{\bf v} \cdot \frac{\partial f}{\partial {\bf v}} = 0.

Now, in Cartesian coordinates, :math:`\dot{\bf v}` is effectively the gravitational force; however, this is not the case
when curvilinear coordinates are in use. Clearly

.. math::

    \frac{d{\bf x}}{dt} = \frac{d{\bf x}}{dx^i}\dot{x}^i = \dot{x}^i {\bf e}_i, \;\text{No sum}.

Furthermore,

.. math::

    \frac{d^2{\bf x}}{dt^2} = \frac{d}{dt}\left[\frac{d{\bf x}}{dx^i}\dot{x}^i\right] =
    \frac{d}{dt}\left[\frac{d{\bf x}}{dx^i}\right]\dot{x}^i + \frac{d{\bf x}}{dx^i} \ddot{x}^i =
    \left[\frac{d^2{\bf x}}{dx^i dx^j}\right]\dot{x}^j\dot{x}^i + \frac{d{\bf x}}{dx^i} \ddot{x}^i

Now, :math:`{\bf e}_i = d{\bf x}/dx^i`, but the coefficient on the first term on the RHS is not as familiar. We define
this to be the **Christoffel Symbol**, which depends on the coordinate system such that

.. math::

    \frac{d^2{\bf x}}{dx^i dx^j} = \Gamma_{ij}^k {\bf e}_k.

We then find that

.. math::

    \boxed{
    \left(\frac{d^2{\bf x}}{dt^2}\right)^k = \ddot{x}^k + \Gamma^k_{ij} \dot{x}^i \dot{x}^j.
    }

Incorporating this into the CBE, we find

.. math::

    \boxed{
    \frac{\partial f}{\partial t} +
    \dot{\bf x} \cdot \frac{\partial f}{\partial {\bf x}} -
    \nabla \Phi \cdot \frac{\partial f}{\partial {\bf v}} -
    {\bf v}^T \Gamma {\bf v} \cdot \frac{\partial f}{\partial {\bf v}}
    = 0.
    }

This is the form of the CBE which is relevant to the models present in Pisces and which must now be "solved" in order to
determine the properties of the distribution function.

The Jeans Equations
''''''''''''''''''''

The CBE in its natural state is not particularly useful. Because it is provides :math:`N` equations with :math:`2N+1` unknowns,
it is not generally solvable. Furthermore, it is generally not possible to solve it directly. We therefore must take a more creative
approach to determine what we need to know about the distribution function. A standard approach is to construct the so-called
**Jeans Equations** by taking velocity-moments of the CBE. Doing so can help us obtain statistics of the distribution function
and therefore determine important properties of the distribution.

To begin, we consider the zeroth-order moments of the CBE. That is

.. math::

    \int_{\mathbb{P}} \frac{\partial f}{\partial t} +
    {\bf v} \cdot \frac{\partial f}{\partial {\bf x}} -
    \nabla \Phi \cdot \frac{\partial f}{\partial {\bf v}} -
    {\bf v}^T \Gamma {\bf v} \cdot \frac{\partial f}{\partial {\bf v}}\; d{\bf p}
    = 0.

The second term is

.. math::

    \int {\bf v} \cdot \frac{\partial f}{\partial {\bf x}} \; d{\bf v} = \frac{\partial}{\partial {\bf x}} \int {\bf v} f \; d{\bf v} =
    \frac{\partial(\nu \bar{\bf v})}{\partial {\bf x}},

where

.. math::

    \nu = \int f \; d{\bf v},

and

.. math::

    \overline{v}^k = \frac{1}{\nu} \int fv^k \; d{\bf v}.

The third term is, upon application of the divergence theorem, simply 0. Finally, the forth term is

.. math::

    \begin{aligned}
    \Gamma^k_{ij} \int v^iv^j \frac{\partial f}{\partial v^k} d{\bf v} &= \Gamma^k_{ij} \left(\int \frac{\partial}{\partial v^k} \left[v^iv^j f\right] - f\frac{\partial}{\partial v^k} \left[v^iv^j\right] d{\bf v}\right)\\
    &= - \Gamma^k_{ij} \int f\frac{\partial}{\partial v^k} \left[v^iv^j\right] d{\bf v}\\
    &= - 2\Gamma^k_{ik} \int v^i f \; d{\bf v}\\
    &= -2\Gamma^k_{ik} \nu \bar{v}^i.
    \end{aligned}

Thus, we arrive at the **0th Order Jeans Equations**:

.. math::

    \boxed{
    \frac{\partial \nu}{\partial t} + \frac{\partial \nu \overline{v}^k}{\partial x^k} + 2\nu\Gamma^k_{ik}\overline{v}^i = 0.
    }

.. note::

    This is a **continuity-like** equation. The final term on the LHS is the "torsion" term which is zero in Cartesian
    coordinates but adds non-zero behaviors in curvilinear coordinate systems.

We can also consider the **1st Order Jeans Equations**; however, this is a yet longer derivation and is stowed away
in a dropdown!

.. dropdown:: 1st Order Jeans Equations

    The first moments (indices :math:`m` and :math:`k`) are

    .. math::

        \int_{\mathbb{P}} v^m \frac{\partial f}{\partial t} +
        v^m v^j\frac{\partial f}{\partial x^j} -
        v^m\nabla^j \Phi  \frac{\partial f}{\partial v^j} -
        v^m \Gamma^k_{ij} v^iv^j  \frac{\partial f}{\partial v^k}\; d{\bf v}
        = 0.

    Introducing the notation

    .. math::

        M^{ij\ldots} = \int v^iv^j\ldots  f \; d{\bf v},

    the first term in the expression may be written

    .. math::

        \int_{\mathbb{P}} v^m \frac{\partial f}{\partial t} d{\bf v} = \frac{\partial M^m}{\partial t}.

    The second term is, likewise, quite simply expressed as

    .. math::

        \int_{\mathbb{P}} v^m v^j \frac{\partial f}{\partial x^j} d{\bf v}  = \frac{\partial M^{mj}}{\partial x^j}.

    The third term is somewhat more complex. Utilizing divergence theorem, we find

    .. math::

        \begin{aligned}
        \nabla^j \Phi \int_{\mathbb{P}} v^m \frac{\partial f}{\partial v^j} \; d{\bf v} &= \nabla^j \Phi \int_{\mathbb{P}} v^m \frac{\partial f}{\partial v^j} \; d{\bf v}\\
        &= \nabla^j \Phi \int_{\mathbb{P}} \frac{\partial}{\partial v^j} \left[fv^m\right] - f \frac{\partial v^m}{\partial v^j} \; d{\bf v}\\
        &= -\nabla^j \Phi \int_{\mathbb{P}} f \delta_j^m \; d{\bf v}\\
        &= -\nu \nabla^m \Phi
        \end{aligned}

    The final term is the most complex by some margin. We will effectively mirror the technique used in the zeroth order derivation;
    however, we now have an additional velocity. Thus, the forth term takes the form

    .. math::

        \begin{aligned}
        \int v^m \Gamma^k_{ij} v^iv^j  \frac{\partial f}{\partial v^k}\; d{\bf v} &=
        \Gamma^k_{ij} \int \frac{\partial}{\partial v^k} \left[v^mv^iv^j f\right] - f\frac{\partial}{\partial v^k} \left[v^mv^iv^j\right]\; d{\bf v}\\
        &= -\Gamma^k_{ij} \int f\frac{\partial}{\partial v^k} \left[v^mv^iv^j\right]\; d{\bf v}\\
        &= -\Gamma^k_{ij} \int f\left[\delta_k^m v^iv^j + \delta_k^i v^mv^j + \delta_k^j v^mv^i\right]\; d{\bf v}\\
        &= -\Gamma^m_{ij} \int f v^iv^j d{\bf v} + -2\Gamma^k_{lk} \int fv^mv^l \; d{\bf v}\\
        &= -\Gamma^m_{ij} M^{ij}  -2\Gamma^k_{lk} M^{ml}.
        \end{aligned}

    We therefore have the general 1st order Jean's equations:

    .. math::

        \boxed{
        \frac{\partial M^m}{\partial t} + \frac{\partial M^{mj}}{\partial x^j} + \nu \nabla^m\Phi + \Gamma^m_{ij} M^{ij}  + 2\Gamma^k_{lk} M^{ml} = 0
        }

    When combined with our 0th order equation, this is yet simpler. The zeroth order is

    .. math::

        \frac{\partial \nu}{\partial t} + \frac{\partial M^k}{\partial x^k} + 2\Gamma^k_{ik}M^i = 0.

    Multiplying by :math:`M^m/\nu`, we have

    .. math::

        \begin{aligned}
        0 &= \frac{M^m}{\nu}\frac{\partial \nu}{\partial t} + \frac{M^m}{\nu}\frac{\partial M^k}{\partial x^k} + \frac{2}{\nu}\Gamma^k_{ik}M^iM^m\\
        &= \frac{M^m}{\nu}\frac{\partial \nu}{\partial t} +  \frac{\partial \left(M^kM^m/\nu\right)}{\partial x^k} - \frac{M^k}{\nu}\frac{\partial M^m}{\partial x^k} + \frac{2}{\nu}\Gamma^k_{ik}M^iM^m
        \end{aligned}

    Subtracting this from the first order equation yields

    .. math::

        \begin{aligned}
        0 &= \frac{\partial M^m}{\partial t} - \frac{M^m}{\nu} \frac{\partial \nu}{\partial t} + \frac{\partial M^{mj}}{\partial x^j} - \frac{1}{\nu} \frac{\partial (M^jM^m/\nu)}{\partial x^j}
        + \frac{M^j}{\nu} \frac{\partial M^m}{\partial x^j} + \Gamma^m_{ij} M^{ij} + 2\Gamma^k_{lk}\left(M^{ml} - \frac{M^m M^l}{\nu} \right) + \nu \nabla^m \Phi
        \end{aligned}

    Letting

    .. math::

        \Sigma^{ml} = M^{ml} - \frac{M^mM^l}{\nu},

    This becomes

    .. math::

        \begin{aligned}
        0 &= \frac{\partial M^m}{\partial t} - \frac{M^m}{\nu} \frac{\partial \nu}{\partial t} + \frac{\partial \Sigma^{mj}}{\partial x^j}
        + \frac{M^j}{\nu} \frac{\partial M^m}{\partial x^j} + \Gamma^m_{ij} M^{ij} + 2\Gamma^k_{lk}\Sigma^{ml} + \nu \nabla^m \Phi\\
        0 &= \nu\frac{\partial (M^m/\nu)}{\partial t} + \frac{\partial \Sigma^{mj}}{\partial x^j}
        + \frac{M^j}{\nu} \frac{\partial M^m}{\partial x^j} + \Gamma^m_{ij} M^{ij} + 2\Gamma^k_{lk}\Sigma^{ml} + \nu \nabla^m \Phi.
        \end{aligned}

    Finally, letting :math:`M^m/\nu = \overline{v_m}` and

    .. math::

        \sigma^{ij} = \frac{1}{\nu} \Sigma^{ij} = \overline{v^iv^j} - \overline{v^i}\cdot \overline{v^j},

    we arrive at the following statement:

    .. math::

        \boxed{
        0 = \nu\frac{\partial \overline{v}^j}{\partial t} + \frac{\partial \left[\nu \sigma^{ij}\right]}{\partial x^i}
        + \nu \overline{v}^i \frac{\partial \overline{v}^j}{\partial x^i} + \nu \nabla^j \Phi + \nu\Gamma^j_{kl} \overline{v^kv^l} + 2\nu\Gamma^m_{nm}\sigma^{jn} .
        }

The 0th order Jeans equation provides a set of :math:`N` equations (index :math:`k`) with :math:`N+1` unknowns (:math:`\nu` + :math:`N \times` zeroth moments).
Likewise, the 1st order Jeans equations do not provide a sufficient number of equations to uniquely determine the distribution. This is true of all orders of Jean's equations and
leads to the inescapable fact that there are multiple distribution functions which are valid for each choice of :math:`\rho({\bf x})`.

To break this degeneracy, we typically constrain the dispersion tensor (:math:`\sigma^{ij}`) to behave a specific way which is convenient.

The Jeans Theorem
''''''''''''''''''''

The **Jeans Theorem** is a central result of the analysis of collisionless systems. To introduce it, we first need the notion of
an **integral**. An integral of motion is a function in phase space :math:`I({\bf x},{\bf p})` such that

.. math::

    \frac{dI}{dt} = \dot{\bf x} \partial_{\bf x} I + \dot{\bf p} \partial_{\bf p} I = 0.

It is quite trivial to show also that :math:`I` satisfying these conditions also solves the CBE! As such, we recognize the following:

.. admonition:: Jeans Theorem

    Any function of the integrals of motion is a solution of the equilibrium collisionless Boltzmann equation.
    Furthermore, any solution of the equilibrium collisionless Boltzmann equation only depends on the phase-space coordinates
    through the integrals of motion.

In many cases where the integrals of motion are known quantities; we can use Jeans Theorem to solve for the distribution.

Virialization Methods
---------------------



Local Maxwellian Approximation
''''''''''''''''''''''''''''''

**Local Maxwellian Approximation** is an approximate procedure for virialization which is very widely applicable
(and therefore useful), but is not a particularly accurate methodology :footcite:p:`kazantzidis_generating_2004`. The basic principle
is that (for certain assumptions about :math:`\sigma_{ij}`) the Jean's equations can be used to uniquely determine statistics
of the distribution function. One then makes an assumption about the structure of the distribution function such that it reproduces those
properties and samples from it. A detailed description of this technique has been described in various places in the literature, most
prominently in :footcite:t:`hernquist_n_body_1993`.




Eddington's Formula for Spherical Symmetry
''''''''''''''''''''''''''''''''''''''''''

There is a special case in which distribution functions can computed exactly (when they exist). In spherically symmetric systems
with no rotation, the only integral of motion is the energy :math:`E`. Thus, the distribution function is a single variable function
of the energy. For notational convenience, we introduce the notation of the relative potential

.. math::

    \Psi = -\Phi + \Phi_0,

for some constant :math:`\Phi_0`. We also define the **relative energy** :math:`\mathcal{E}` as

.. math::

    \mathcal{E} = -E + \Phi_0 = \Psi - \frac{1}{2}v^2.

Using these definitions, we can derive the very useful **Eddington's Formula**:

.. math::

    f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2} \left[\int_0^\mathcal{E} d\Psi \; \frac{1}{\sqrt{\mathcal{E}-\Psi}} \frac{d^2\rho}{d\Psi^2} + \frac{1}{\sqrt{\mathcal{E}}}\left.\frac{d\rho}{d\Psi}\right|_{\Psi=0}\right]


.. dropdown:: Derivation

    Assuming that the distribution function is :math:`f(\mathcal{E})`, we can write the density as

    .. math::

        \begin{aligned}
        \rho(r) &= \int f(r,{\bf v}) d{\bf v}\\
                &= 4\pi \int v^2 f(r,v) dv\\
        \end{aligned}

    To express the integral in terms of the energy :math:`\mathcal{E}`, we let :math:`d\mathcal{E} = - v \; dv` and
    :math:`v^2 = 2(\Psi - \mathcal{E})`. Thus,

    .. math::

        \rho(r) = 4\pi \int_0^\Psi \sqrt{2(\Psi - \mathcal{E})} f(\mathcal{E})\; d\mathcal{E}.

    Because the distribution is spherical, :math:`\Phi` (and :math:`\Psi`) increase monotonically with radius. Thus, we can
    write :math:`r(\Psi)` and state that

    .. math::

        \frac{1}{\pi \sqrt{8}} \rho(\Psi) = 2 \int_0^{\Psi} f(\mathcal{E}) \sqrt{\Psi - \mathcal{E}}  \; d\mathcal{E}.

    Taking the derivative on either side, we have the expression

    .. math::

        \frac{1}{\sqrt{8}\pi} \frac{d\rho(\Psi)}{d \Psi} = \int_0^\Psi \frac{f(\mathcal{E})}{\sqrt{\Psi - \mathcal{E}}} d\mathcal{E}.

    This is an Abel integral equation, which means it can be inverted in the form

    .. math::

        f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2} \left[\int_0^\mathcal{E} d\Psi \; \frac{1}{\sqrt{\mathcal{E}-\Psi}} \frac{d^2\rho}{d\Psi^2} + \frac{1}{\sqrt{\mathcal{E}}}\left.\frac{d\rho}{d\Psi}\right|_{\Psi=0}\right]

The Eddington formula can be used to construct virialized particle velocities in spherical coordinates. Because the
method is numerically quite tractable, it is almost always to best choice when working with spherically symmetric systems.
