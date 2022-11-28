.. _geomstats_landingpage:

Geomstats
=========

.. image:: geomstats_logo.jpg
  :width: 160
  :alt: Geomstats Logo

**Geomstats** is an open-source Python package for computations, statistics, and machine learning on
nonlinear manifolds. Data from many application fields are elements of manifolds. For instance,
the manifold of 3D rotations SO(3) naturally appears when performing statistical learning on 
articulated objects like the human spine or robotics arms. Likewise, shape spaces modeling biological shapes or other natural shapes 
are manifolds. Additional examples are introduced in Geomstats `paper <https://arxiv.org/abs/2004.04667>`_. Geomstats' `source code <https://github.com/geomstats/geomstats>`_ is freely available on GitHub.

.. figure:: geomstats_examples.jpg
   :alt: natural shapes
   :target: https://geomstats.github.io/notebooks/11_real_world_applications__cell_shapes_analysis.html
   :class: with-shadow
   :width: 1000px

   **Figure**: Shapes in natural sciences can be represented as data points on "manifolds". Images credits: Self Reflected, `Greg Dunn Neuro Art <www.gregadunn.com>`_, British Art Foundation, Ashok Prasad, Matematik Dunyasi, Gabriel Pérez.

Computations and statistics on manifolds require special tools of
`differential geometry <https://en.wikipedia.org/wiki/Differential_geometry>`_. Computing
the mean of two rotation matrices :math:`R_1, R_2` as :math:`\frac{R_1 + R_2}{2}` does not
generally give a rotation matrix. Statistics for data on manifolds need to be extended to
"geometric statistics" to perform consistent operations.

Objectives
----------

Geomstats is here to fulfill four objectives:

1. provide educational support to learn "hands-on" differential geometry and geometric statistics, through its examples and visualizations.
2. foster research in differential geometry and geometric statistics by providing operations on manifolds to gain intuition on results of a research paper;
3. democratize the use of geometric statistics by implementing user-friendly geometric learning algorithms using Scikit-Learn API; and
4. provide a platform to share learning algorithms on manifolds.

Design
------

Geomstats is organized into two main modules: `geometry` and `learning`.

The module `geometry <https://github.com/geomstats/geomstats/tree/master/geomstats/geometry>`_ implements concepts from differential geometry,
such as manifolds and Riemannian metrics. The module `learning <https://github.com/geomstats/geomstats/tree/master/geomstats/learning>`_ implements statistics and learning algorithms for data
on manifolds, such as supervised and unsupervised learning techniques.

.. figure:: conn_parallel_vector_field.jpeg
   :alt: parallel vector field
   :target: https://github.com/geomstats/geomstats/blob/master/notebooks/01_foundations__manifolds.ipynb
   :class: with-shadow
   :width: 1000px

   **Figure**: Parallel transport of a vector X (pink) along a geodesic (green) on the manifold M, e.g. representing a deformation's force acting on the time evolution of an organ shape. Image credits: `Adele Myers <https://ahma2017.wixsite.com/adelemyers>`_.

The code is object-oriented and follows Scikit-Learn's API. The operations are vectorized for batch computation and provide
support for different execution backends --- namely NumPy, PyTorch, Autograd and TensorFlow.

Learn More
----------

To learn how to use `geomstats`, visit :ref:`first_steps`. To contribute to `geomstats` visit :ref:`contributing`. 
To learn more about differential geometry and manifolds, visit :ref:`explanation`. To find more advanced examples, 
visit :ref:`tutorials`.


Geomstats Documentation Overview
===================================

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started/index
   explanation/index
   tutorials/index
   contributing/index
   api/index
   Roadmap <roadmap>
   Governance <governance>
   Google Season of Docs <gsod>
   Hackathons <hackathons>

**Version**: |version|

**Downloadable versions of the documentation**:
`PDF Version <https://hal.inria.fr/hal-02536154v1/document>`_

Welcome! this the documentation summary page for the Geomstats project.
**Geomstats** is an open-source Python package for computations, statistics, 
and machine learning on nonlinear manifolds. Below is a complete overview
of the main sections in the documentation.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: ./_static/index-images/index_getting_started.svg

    Getting Started
    ^^^^^^^^^^^^^^^

    To learn how to use `geomstats` as a beginner, start with this section. 
    It contains an introduction, first steps and code examples to help you
    understand the project.

    +++

    .. link-button:: getting_started/index
            :type: ref
            :text: To the getting started guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index-images/index_user_guide.svg

    Tutorials
    ^^^^^^^^^^

    The tutorials provide in-depth guides on practical methods and real applications of 
    the `geomstats` package.

    +++

    .. link-button:: tutorials
            :type: ref
            :text: To the tutorials
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index-images/index_api.svg

    API reference
    ^^^^^^^^^^^^^

    The reference guide has a detailed break down of the public API functions of the
    `geomstats` project. This includes classes, their attributes, and explanation of 
    what the methods do.

    +++

    .. link-button:: api/index
            :type: ref
            :text: To the API reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index-images/index_contribute.svg

    Contributor's Guide
    ^^^^^^^^^^^^^^^^^^^

    If you are curious about the internals of the project, or want to improve
    the project by reporting an issue, submiting a patch or even triaging, this 
    section details the development process..

    +++

    .. link-button:: contributing
            :type: ref
            :text: To the contributor's guide
            :classes: btn-block btn-secondary stretched-link
