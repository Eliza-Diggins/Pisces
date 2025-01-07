.. _sampling:

===============================
Sampling Particles From Models
===============================

Every :py:class:`~pisces.models.base.Model` instance is composed of **fields**, which are discretely defined grids
of physical data about the system. In some cases, these fields are sufficient for scientific objectives. Many
simulation codes (grid-based codes) only need the data of the model to be interpolated onto a native simulation grid
to run. However, in many scenarios, it is necessary to sample particles from these fields and use these particles for
computations of interest. Pisces natively supports the generation of particle datasets from every model.

Overview
--------

To create a particle dataset (:py:class:`~pisces.particles.base.ParticleDataset`) from a model, there are three primary
steps in the process:

1. **Particle Sampling**: Determine where to place the particles in physical space.
2. **Virialization**: For collisionless components (e.g., dark matter), determine the particle velocities to
   maintain stability.
3. **Field Interpolation**: Assign values to any necessary particle fields by interpolating the progenitor model fields
   to the positions of the particles.

Each of these steps is a non-trivial exercise in computational methods.

Sampling Particles From Fields
------------------------------

Particle sampling can be effectively framed as follows:

.. admonition:: The General Question

    Let :math:`q_1, \ldots, q_d` be a :math:`d`-dimensional coordinate system spanning :math:`\mathbb{R}^d`. Let
    :math:`f: \mathbb{R}^d \to \mathbb{R}` be a field dependent on a subset of the available
    coordinates :math:`q_{\alpha_1}, q_{\alpha_2}, \ldots, q_{\alpha_N}`.

    We wish to construct an ensemble of points :math:`X \subset \mathbb{R}^d` such that

    .. math::

        \forall \mathbf{x} \in X,\, P(q_i \le x_i \le q_i + \delta q_i) \propto f(\mathbf{q}) J(\mathbf{q}),

    where :math:`J(\mathbf{q})` is the Jacobian.

.. dropdown:: Mathematical Note: The Jacobian

    If :math:`f(\mathbf{q})` represents a density field (e.g., dark matter or gas density), then it is natural
    for the PDF of particle positions to relate to the density. However, the involvement of the Jacobian may
    be less intuitive. More formally, the probability of a particle being in a neighborhood :math:`N` (of size
    :math:`dV`) around a point :math:`\mathbf{q}` is given by the ratio of the mass inside that neighborhood to the total mass:

    .. math::

        P(\mathbf{x} \in N(\mathbf{q})) = \frac{f(\mathbf{q}) dV}{\int_D f(\mathbf{q}) dV}.

    Since :math:`dV = J(\mathbf{q}) \prod_i dq^i`, it follows that:

    .. math::

        P(\mathbf{x} \in N(\mathbf{q})) = \frac{f(\mathbf{q}) J(\mathbf{q}) d^d q}{\int_D f(\mathbf{q}) J(\mathbf{q}) d^d q}.

    Thus, the PDF depends on the Jacobian.

Approach
++++++++

1. Sample particles from the distribution proportional to :math:`f(\mathbf{q}) J(\mathbf{q})`.
2. Convert the particle positions to Cartesian coordinates.

The primary challenge lies in the first step. Different methods for sampling particles suit different problems, requiring
an adaptive approach for Pisces.

Approach 1: Fully Separable Fields
'''''''''''''''''''''''''''''''''''

The simplest sampling scenario is as follows:

.. admonition:: Inverse Transform Requirements

    Let :math:`q_1, \ldots, q_d` be a :math:`d`-dimensional coordinate system spanning :math:`\mathbb{R}^d`. Let :math:`f: \mathbb{R}^d \to \mathbb{R}` be
    a field dependent on a subset of the available coordinates :math:`q_{\alpha_1}, q_{\alpha_2}, \ldots, q_{\alpha_N}`.

    Assume the Jacobian of the coordinate system is:

    .. math::

        J(\mathbf{q}) = \prod_l J_l(q^l),

    and :math:`\forall i \le N`, the field takes the form:

    .. math::

        f(\mathbf{q}) = \prod_l f_{\alpha_l}(q^{\alpha_l}).

In this scenario, the probability distribution simplifies:

.. math::

    P(\mathbf{q}) = \frac{f(\mathbf{q}) J(\mathbf{q}) d^d q}{\int_D f(\mathbf{q}) J(\mathbf{q}) d^d q} = \prod_l P(q^l),

where

.. math::

    P(q^l) = \frac{f_l(q^l)J_l(q^l) dq^l}{\int_{D^l} f_l(q^l)J_l(q^l) dq^l}

is a 1-dimensional distribution function.

.. tip::

    When :math:`f` is independent of :math:`q^l`, the distribution depends only on the Jacobian, which is analytically known.

This reduction breaks a complex :math:`d`-dimensional distribution into :math:`d` 1-dimensional distributions, which can
be independently sampled to construct the particle ensemble.

.. admonition:: Inverse Transform Sampling

    In cases where the sampling problem reduces to a single dimension,
    `inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_ is a straightforward and
    effective method.

    Given a probability density function (PDF) :math:`f(x)`, the first step is to compute its
    cumulative distribution function (CDF):

    .. math::

        F(x) = \int_{x_\text{min}}^x f(\xi) \, d\xi, \quad F(x) \in [0, 1],

    where :math:`x_\text{min}` is the lower bound of the domain of :math:`f(x)`. The CDF :math:`F(x)` represents
    the cumulative probability up to the value :math:`x`, normalized to the range :math:`[0, 1]`.

    To sample particle positions:

    1. Generate random variates, :math:`u`, uniformly distributed in :math:`[0, 1]`.
    2. Compute the corresponding particle positions by applying the inverse of the CDF:

       .. math::

           x = F^{-1}(u), \quad u \sim U[0, 1].

    Here, :math:`F^{-1}(u)` is the value of :math:`x` such that :math:`F(x) = u`. This transforms the uniform
    random variates into positions distributed according to the PDF :math:`f(x)`.

Approach 2: Rejection Sampling
''''''''''''''''''''''''''''''

When separability is not available (typically because :math:`f({\bf q})` depends on many field variables), we require a more
sophisticated method by which to perform the sampling. In these cases, we rely on `rejection sampling <https://en.wikipedia.org/wiki/Rejection_sampling>`_
to draw samples from the complex PDF functions.

.. dropdown:: Rejection Sampling Theory [Optional]

    The general principle of rejection sampling is to draw samples from some (known but un-normalized) distribution
    :math:`f: \mathbb{R}^k \to \mathbb{R}` by instead considering the more general problem of sampling uniformly from
    :math:`\mathbb{R}^{k+1}` and then accepting samples only if the fall below the surface generated by :math:`f`.

    More formally, assume that a random variate :math:`X` has (unknown) distribution function :math:`f:\mathbb{R}^k \to \mathbb{R}`
    proportional to a known likelihood function :math:`\tilde{f}`. Furthermore, let :math:`Y` be another random variate with
    a known probability density :math:`g: \mathbb{R}^k \to \mathbb{R}`. Finally, assume that :math:`\exists M \in \mathbb{R}` such
    that,

    .. math::

        \forall {\bf x} \in \mathbb{R}^k \, \text{s.t.}\, f({\bf x}) \neq 0, f({\bf x}) \le M g({\bf x}).

    .. hint::

        This also implies that there is some :math:`M'` such that the above holds for :math:`\tilde{f}` instead of :math:`f`.

    .. note::

        Recognize that this requires that :math:`\tilde{f}(Y)/(M' g(Y)) \in [0,1]`!

    If you now sample pairs of points :math:`(x,v)` such that :math:`x \sim Y`, and :math:`v = u Mg(x)` where :math:`u \sim U[0,1]`
    you produce a uniform sample over the subgraph of :math:`Mg(x)`. Thus, if samples are accepted only when

    .. math::

        v \le f(x) \implies u \le \frac{f(x)}{Mg(x)},

    then :math:`x \sim X`.

