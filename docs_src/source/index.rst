.. lys_em documentation master file, created by
   sphinx-quickstart on Thu Mar 27 13:07:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lys_em: JAX-accelerated Electron Microscopy Simulation
========================================================

*lys_em* is an object-oriented Python-based library designed for the high-performance simulation
and analysis of electron microscopy (EM) images and diffraction patterns.

Overview
--------

`lys_em` provides a versatile, object-oriented framework for simulating various EM modalities.
At its core, the library utilizes the **multislice method** to provide accurate physical modeling of electron-matter interactions.

Getting Started
-----------------

Source code of *lys_em* is opened in GitHub (https://github.com/a-tock/lys_em).

To use *lys_em*, go to :doc:`install` and try :doc:`tutorial`.

Characteristics
-----------------

* **Generalized Object-Oriented Design**
    A modular architecture that allows for intuitive setup of potentials and imaging conditions.

* **High-Performance Computing via JAX**
    By leveraging the `JAX` backend, `lys_em` provides:

    * **Speed and Scalability:**
      Computational tasks are highly optimized and can be seamlessly parallelized across CPUs, GPUs, and TPUs.
    * **Automatic Differentiation:**
      The simulator is fully differentiable. This enables high-speed, gradient-based optimization using experimental data
      across a wide range of structural and instrumental parameters.

Future Vision
-------------

We are actively developing new features to enhance the library's utility:

* **PRISM Algorithm:** Implementation of the PRISM (Position Resolved Iterative Sub-diffraction Microscopy) method for accelerated STEM simulations.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   tutorial
   api
   contributing
