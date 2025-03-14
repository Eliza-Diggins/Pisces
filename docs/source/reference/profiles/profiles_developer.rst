.. _profiles-developer:

========================
Building Custom Profiles
========================

A relatively common need for users of Pisces is to write custom profiles that can be used in whichever modeling
procedure is relevant to your scientific need. Because of this, Pisces has been written with the explicit intention of
making it easy to write custom profiles. In this document, we'll provide a detailed overview of writing custom profiles.

Scaffolding
-----------

As our primary example, we'll rebuild the famous `NFW profile <https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile>`_ from scratch.
The first thing to do is create a subclass of the :py:class:`~pisces.profiles.base.Profile` class to build our custom profile from:

.. code-block:: python

    from pisces.profiles.base import Profile

    class NFWProfile(Profile):
        """
        Custom NFW profile implementation.
        """
        # Custom code will go here.

Now, the profile needs to know about which axes and parameters are present. To specify these, we include the
:py:attr:`~pisces.profiles.base.Profile.AXES` and :py:attr:`~pisces.profiles.base.Profile.DEFAULT_PARAMETERS` class attributes.

.. code-block:: python

    from pisces.profiles.base import Profile
    from unyt import unyt_quantity #(allows us to use units)

    class NFWProfile(Profile):
        """
        Custom NFW profile implementation.
        """
        AXES = ['r']
        AXES_UNITS = {'r':'kpc'}
        DEFAULT_PARAMETERS = {
            "rho_0": unyt_quantity(1.0,'Msun/kpc**3'),
            "r_s": unyt_quantity(1.0,'kpc'),
        }

This is all the scaffolding necessary to create the basic profile structure. Now we just need to actually add the formula.

Adding the Formula
------------------

The NFW profile is

.. math::

    \rho = \rho_0 \left(\frac{r}{r_s}\right)^{-1} \left(1+\frac{r}{r_s}\right)^{-2}.

To implement this in our profile, we need to specify the ``_profile`` method in the class. The ``_profile`` method should
take the independent variables (``r``) and the parameters (``rho_0, r_s``) as arguments and return a **sympy expression**.

.. note::

    It's important that this is done with the various ``sympy`` mathematics functions, not ``numpy``. We convert everything
    to ``numpy`` internally, but your code will break if you try to pass symbols through numpy functions.

.. code-block:: python

    from pisces.profiles.base import Profile
    from unyt import unyt_quantity #(allows us to use units)
    import sympy as sp

    class NFWProfile(Profile):
        """
        Custom NFW profile implementation.
        """
        AXES = ['r']
        AXES_UNITS = {'r':'kpc'}
        DEFAULT_PARAMETERS = {
            "rho_0": unyt_quantity(1.0,'Msun/kpc**3'),
            "r_s": unyt_quantity(1.0,'kpc'),
        }

        def _profile(r,rho_0=1,r_s=1):
            return rho_0*((r/r_s)**(-1))*((1+(r/r_s))**(-2))

That’s it! With this in place, the metaclass takes care of generating symbolic axes and parameters, constructing
a fully symbolic expression, and then creating the callable function automatically when you instantiate the profile.

Symbolic Expressions
--------------------
A powerful feature in Pisces is the ability to register class-level symbolic expressions on your profile. These are
symbolic manipulations or derived quantities that are relevant to every instance of the profile.
A common example might be computing an analytical derivative or cumulative integral. By defining them at the class
level, you avoid recalculating the symbolic manipulation for each instance, and can then evaluate
or further manipulate those expressions on demand.

To do this, use the :py:func:`~pisces.profiles.base.class_expression` decorator on a static method in your class.
For example, suppose we want to add a class-level derivative of the NFW profile with respect to the radial coordinate.
We can do the following:

.. code-block:: python

    from pisces.profiles.base import Profile, class_expression
    from unyt import unyt_quantity
    import sympy as sp

    class NFWProfile(Profile):
        """
        Custom NFW profile implementation with a derivative expression.
        """
        # Do not treat this class as abstract
        _is_parent_profile: bool = False

        AXES = ['r']
        AXES_UNITS = {'r':'kpc'}
        DEFAULT_PARAMETERS = {
            "rho_0": unyt_quantity(1.0,'Msun/kpc**3'),
            "r_s": unyt_quantity(1.0,'kpc'),
        }

        @staticmethod
        def _profile(r, rho_0=1, r_s=1):
            return rho_0 * (r / r_s) ** (-1) * (1 + r / r_s) ** (-2)

        @class_expression(name="d_drho_dr", on_demand=True)
        @staticmethod
        def _nfw_derivative(axes, params, expression):
            """
            Symbolically differentiate the NFW expression w.r.t. r.

            Parameters
            ----------
            axes : List[sympy.Symbol]
                Symbolic axes of this profile, e.g. [r].
            params : Dict[str, sympy.Symbol]
                Dictionary mapping parameter names (e.g. 'rho_0', 'r_s') to Sympy symbols.
            expression : sympy.Basic
                The main NFW profile expression: rho(r).

            Returns
            -------
            sympy.Basic
                d(rho)/dr in symbolic form.
            """
            r = axes[0]
            # We simply return the derivative wrt the first axis (r).
            return sp.diff(expression, r)

Here’s what is happening in the code above:

- We decorate the method with :py:func:`~pisces.profiles.base.class_expression(name="d_drho_dr", on_demand=True)`. This determines
  the name under which the expression is registered (``"d_drho_dr"``). ``on_demand=True`` specifies that we will not
  immediately compute this expression at class creation time. Instead, Pisces will store a reference to the method, and
  only compute (symbolically) when a user first requests it.

- The decorated method must accept exactly:

  - axes: The list of Sympy symbols corresponding to :py:attr:`~pisces.profiles.base.Profile.AXES` (in this case, just ``[r]``).
  - params: A dictionary that maps parameter names to their Sympy symbols (e.g., ``{"rho_0": rho_0_symbol, "r_s": r_s_symbol}``).
  - expression: The class’s main symbolic expression (the :py:attr:`~pisces.profiles.base.Profile.profile_expression`), still un‐substituted.
