.. role::  raw-html(raw)
    :format: html

.. _getting_started:

Quickstart Guide
----------------

Welcome to **Pisces**! On this page, you can find various guides for setting up the software and getting started
with model building.

Installation and Setup
''''''''''''''''''''''

Pisces is written primarily in Python and can be installed just like any other package using ``pip`` or ``conda``. To install
using ``pip``, simply use the

.. code-block:: bash

    $pip install pisces

command and the code will begin downloading and compiling. If you want to install the code directly from source (via github),
you should still use ``pip``; however, the command sequence is now

.. code-block:: bash

    $pip install git+https://github.com/eliza-diggins/pisces

Once the code has compiled you should be able to import it just like any other python package using

.. code-block:: python

    import pisces as pi

.. raw:: html

   <hr style="height:2px;background-color:black">

Pisces Introduction
''''''''''''''''''''

.. warning::

    This section is still being written.

More Info
'''''''''
Once you've covered the information on this page, you'll find a lot more detail about everything Pisces has to offer
elsewhere in the documentation. We maintain a :ref:`reference` page with a lot of in-depth guides to various components of the
code base and also a :ref:`examples` library that can be used to retrofit existing code for your needs.
