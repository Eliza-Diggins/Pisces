.. _geometry_theory:

Coordinate Systems Theory
-------------------------

All of the coordinate systems in Pisces are *orthogonal coordinate systems* and are managed under the
:py:class:`~pisces.geometry.base.CoordinateSystem` class. An orthogonal coordinate system in Pisces is
a mathematical structure that maps points from a :math:`d`-dimensional Euclidean space, :math:`\mathbb{R}^d`, into another
Euclidean space (effectively, :math:`\mathbb{R}^d` again) by establishing a system of coordinates that are orthogonal at
every point. This structure facilitates various vector calculus operations (e.g., dot product, cross product, gradient,
divergence, curl) by taking advantage of the orthogonality of the basis vectors in the coordinate system.

Definition of the Coordinate Map
++++++++++++++++++++++++++++++++

An orthogonal coordinate system is defined by a set of coordinates :math:`q^1, q^2, \dots, q^d` and a mapping function
:math:`X: \mathbb{R}^d \to \mathbb{R}^d`, which associates each point in the coordinate space with a point in Cartesian space.
This map is invertible, continuous, and differentiable, with the condition that

.. math::

    \frac{\partial^2 X}{\partial q^k \partial q^j} \propto \delta_{kj}.

This orthogonality ensures that each pair of coordinate curves meets at a right angle, greatly simplifying geometric and
differential computations.

Covariant and Contravariant Bases
+++++++++++++++++++++++++++++++++

At each point in the coordinate system, two distinct bases can be defined, both having basis vectors parallel to the
coordinate directions: the **covariant basis** and the **contravariant basis**.

1. **Covariant Basis** :math:`(\mathbf{e}_i)`:
   The covariant basis is given by the set of partial derivatives of the mapping function with respect to each coordinate:

   .. math::

       \mathbf{e}_i = \frac{\partial X}{\partial q^i}.

   These basis vectors are aligned with the coordinate axes and may have different magnitudes, represented by scale factors
   :math:`h_i = \|\mathbf{e}_i\|`, which are also known as **Lame coefficients**.

   .. hint::

       In Pisces, these are referred to as the Lame Coefficients, and they play a critical role in our implementation of
       coordinate systems. Effectively, knowing the Lame Coefficients provides all the information necessary to carry
       out any computation.

2. **Contravariant Basis** :math:`(\mathbf{e}^i)`:
   The contravariant basis vectors form the dual basis to the covariant vectors. They are defined such that they satisfy the
   duality relationship:

   .. math::

       \mathbf{e}^i \cdot \mathbf{e}_j = \delta^i_j,

   where :math:`\delta^i_j` is the Kronecker delta, which is 1 if :math:`i = j` and 0 otherwise. The contravariant basis is
   useful for defining vector components that are reciprocally aligned to the covariant directions. Because we are working
   in orthogonal coordinates, these contravariant vectors (denoted with up indices) are simply

   .. math::

       \mathbf{e}^i = \frac{1}{h_i} \hat{e}_i = \frac{1}{h_i^2} \mathbf{e}^i, \quad \text{which implies} \quad \mathbf{e}^i \mathbf{e}_i = 1.

The unit basis vectors can be obtained by normalizing the covariant and contravariant vectors with their respective scale factors.

.. note::

    In many methods in Pisces, you may see the ``basis='unit'`` keyword argument. This refers to the basis in which a vector
    field is specified or in which to perform a given calculation. In specific contexts, this choice is critical to ensure
    that you get the answer you want.

The Inner Product
+++++++++++++++++++++++++++++++++

Formally, given any vector space :math:`V` over a field :math:`\mathbb{F}`, there exists a set of maps :math:`M` such that

.. math::

    M = \{ f: V \to \mathbb{F} \} \quad \text{for all linear } f.

These are the so-called homomorphisms between :math:`V` and its field :math:`\mathbb{F}`. Such a set is called the *dual space*
of the coordinate system, and it inherits a basis from :math:`V`. Let :math:`e_i` be a basis of :math:`V`, then the dual basis :math:`e^i` is
defined implicitly such that

.. math::

    e^i(e_j) = \delta_{ij}.

In this case, orthogonal coordinates provide such linear maps as vectors, and so we get the notion of covariant and contravariant vectors.

In an orthogonal coordinate system, the **dot product** between two vectors :math:`\mathbf{A}` and :math:`\mathbf{B}` with
components :math:`A^i` and :math:`B^i` in the contravariant basis is defined as:

.. math::

    \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^d h_i^2 A^i B^i,

where :math:`h_i` are the **Lame coefficients**, which represent the scale factors associated with each coordinate direction
and capture the "stretching" of space in that direction.

Differential Operators: Gradient, Divergence, and Curl
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Orthogonal coordinates simplify the definitions of differential operators:

1. **Gradient** :math:`\nabla \Phi`:
   The gradient of a scalar field :math:`\Phi` in orthogonal coordinates is given by:

   .. math::

       \nabla \Phi = \sum_{i=1}^d \frac{1}{h_i} \frac{\partial \Phi}{\partial q^i} \mathbf{e}^i.

2. **Divergence** :math:`\nabla \cdot \mathbf{F}`:
   The divergence of a vector field :math:`\mathbf{F}` with components :math:`F^i` in orthogonal coordinates is:

   .. math::

       \nabla \cdot \mathbf{F} = \frac{1}{h_1 h_2 \dots h_d} \sum_{i=1}^d \frac{\partial}{\partial q^i} \left( h_1 h_2 \dots h_d \, F^i \right).

3. **Curl** :math:`\nabla \times \mathbf{F}`:
   In three-dimensional orthogonal coordinates, the curl of a vector field :math:`\mathbf{F}` is given by:

   .. math::

       \nabla \times \mathbf{F} = \frac{1}{h_1 h_2 h_3} \begin{vmatrix} h_1 \mathbf{e}_1 & h_2 \mathbf{e}_2 & h_3 \mathbf{e}_3 \\
       \frac{\partial}{\partial q^1} & \frac{\partial}{\partial q^2} & \frac{\partial}{\partial q^3} \\
       F^1 & F^2 & F^3 \end{vmatrix}.

Symmetry
--------

In many physical cases, a specific coordinate system is accompanied by a given symmetry. There are many ways to describe
symmetry mathematically; the most common approach being via groups of invariant transformations. For the use of Pisces, it
is sufficient to let a particular symmetry :math:`\mathcal{S}` in a given coordinate system be a set of coordinate

.. math::

    \mathcal{S} = \{a, \; a \in \alpha\}, \alpha \subset \{1,\cdots,N\}

for an :math:`N` dimensional coordinate system. A field with the specific symmetry :math:`\mathcal{S}`, :math:`\phi` is, by
definition, invariant under any deviation in one of the symmetry axes:

.. math::

    \forall k \in \mathcal{S},\; \frac{\partial \phi}{\partial q^k} = 0,

In many cases, a given symmetry is preserved when an operation is performed on :math:`\phi`; however, in other cases, the
same operation may break a particular component of the symmetry.

.. hint::

    This is effectively because the Lame Coefficient can be functions of all of the coordinates and
    thereby introduce dependence in particular cases.

Effects of Operations on Symmetry
+++++++++++++++++++++++++++++++++

Whether an operation preserves or breaks symmetry often depends on the nature of the operation and the structure of the
coordinate system. To be precise, the Lame Coefficients :math:`h_i` of a particular coordinate system are scalar fields
that may exhibit symmetry in the same sense as other fields. Thus, the symmetry of an operation typically depends both on
the symmetry of the field and the symmetry of the Lame coefficients. In the following, let :math:`S_i` denote the symmetry
set of each Lame coefficient :math:`h_i`, and let the coordinate system be :math:`d`-dimensional. Let :math:`S^0` be the
**universal symmetry** which is symmetric in all of the :math:`d` dimensions. Thus

.. math::

    S^0_d = \{1, \cdots d\}.


1. **Partial Differentiation**:
   The partial derivative operator :math:`\partial_k` will either preserve or increase the symmetry of a field. If :math:`\phi`
   is a field with symmetry set :math:`\mathcal{S}` (i.e., :math:`\frac{\partial \phi}{\partial q^k} = 0` for all :math:`k \in \mathcal{S}`),
   taking a partial derivative :math:`\partial_i \phi` along a direction not in :math:`\mathcal{S}` will preserve
   the symmetry already present. Conversely, if :math:`\partial \phi / \partial q^i = 0` for some
   :math:`i`, differentiation in that direction will make the symmetry universal. Thus,

   .. math::

        \partial_k \mathcal{S} = \begin{cases}\mathcal{S},&k \in \mathcal{S}\\S^0,&k\notin \mathcal{S}.\end{cases}

2. **Gradient**:

    The gradient may also interfere with a particular symmetry. The :math:`k`-th element of the gradient is

    .. math::

        \nabla_k \phi = e^k \partial_k \phi,

    Thus, in the contravariant basis, gradient operates on symmetries the same way that a partial derivative does. In the
    unit and covariant bases, this is not the case. Instead,

    .. math::

        \nabla_k \phi = \hat{e}^k \frac{\partial_k \phi}{h_k},

    and the symmetry then becomes

    .. math::

        \nabla_k \mathcal{S} = \mathcal{S} \setminus S_k.

3. **Divergence**:
   The divergence of a vector field :math:`\mathbf{F}` in orthogonal coordinates is defined by:

   .. math::

       \nabla \cdot \mathbf{F} = \frac{1}{h_1 h_2 \dots h_d} \sum_{i=1}^d \frac{\partial}{\partial q^i} \left( h_1 h_2 \dots h_d \, F^i \right).

   The divergence operator preserves symmetry if each component :math:`F^i` and the Lame coefficients respect the symmetry
   of the field. However, divergence may introduce coordinate dependencies due to the term :math:`h_1 h_2 \dots h_d`, which
   can vary across coordinate directions. Specifically, if one of the Lame coefficients :math:`h_i` varies along an axis in
   :math:`\mathcal{S}`, the divergence operation will introduce dependence along that axis and break symmetry.

4. **Curl**:
   In three-dimensional coordinates, the curl of a vector field :math:`\mathbf{F}` is defined by:

   .. math::

       \nabla \times \mathbf{F} = \frac{1}{h_1 h_2 h_3} \begin{vmatrix} h_1 \mathbf{e}_1 & h_2 \mathbf{e}_2 & h_3 \mathbf{e}_3 \\
       \frac{\partial}{\partial q^1} & \frac{\partial}{\partial q^2} & \frac{\partial}{\partial q^3} \\
       F^1 & F^2 & F^3 \end{vmatrix}.

   Symmetry is preserved in the curl operation if the components of :math:`\mathbf{F}` and the Lame coefficients respect
   the symmetry along each axis. For example, a field symmetric about the :math:`z`-axis would retain this symmetry after
   a curl operation if :math:`h_x` and :math:`h_y` are constants or functions of :math:`z` alone. However, if any Lame
   coefficient introduces dependence on :math:`x` or :math:`y`, the curl operation will break symmetry along the :math:`z`-axis.

.. hint::

    Symmetry breaking often arises due to the dependence of the Lame coefficients on the coordinates. If the Lame
    coefficients vary with respect to coordinates that would otherwise be in a symmetry set, operations involving
    these coefficients can introduce dependencies that disrupt the symmetry of the resulting field.

In summary, the symmetry properties of an operation depend on both the symmetry of the field and the symmetry of the Lame
coefficients. Understanding how these factors interact is essential for ensuring that specific symmetries are maintained
through various operations in Pisces.