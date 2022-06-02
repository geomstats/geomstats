.. _roadmap:

=======
Roadmap
=======

Purpose of this document
------------------------

This roadmap lists general directions that core contributors are interested
to see developed in Geomstats. The fact that an item is listed here is in
no way a promise that it will happen, as resources are limited. Rather, it
is an indication that help is welcomed on this topic.

This document is inspired from Scikit-Learn's roadmap.

Statement of purpose: Geomstats in 2022
---------------------------------------

Four years after the inception of Geomstats in 2018, the
world of geometric statistics has witnessed some key changes:

* Computational Geometry: Python packages to perform optimizations or learning on manifolds.
* Computational Mathematics: New packages in computational mathematics have emerged, together 
with an increase of our contributors' pool, featuring PhD students in applied mathematics that are interested in open-source.
* Geometric Deep learning: a fast-growing field, partially supported by the success of graph neural networks.

Geomstats is very popular in practice for computations and canonical
machine learning techniques on data belonging to manifolds, such as shape data, 
particularly for applications in experimental science and specifically in biomedical fields. 

While some Geomstats modules are mature, other are experiencing a fast growth and frequent changes, with an increasing number of new contributors
and an expanding user base. It is costly to maintain the existing modules, and organize the development of new ones.

**Our main goals are to**:

1. continue maintaining a high-quality, well-documented collection of canonical tools for computations, statistics and machine learning for data on manifolds,
2. re-organize and re-design the documentation website,
3. re-organize the architecture of the learning module, with a scikit-learn inspired design with Mixins,
4. develop the shapes module and transfer techniques to communities in applied sciences including biomedical fields,
5. develop the information geometry module,
6. develop the stratified geometry module.

Issues corresponding to these goals can be found under their corresponding GitHub's tag, e.g. the `shapes tag
<https://github.com/geomstats/geomstats/labels/shapes>`_ for goal 4.

Detailed Goals
--------------

The list is numbered not as an indication of the order of priority, but to
make referring to specific points easier. Please add new entries only at the
bottom of each goal. We try to keep the document up to date as we work on these issues.


1. Maintenance

   * Refactor testing infrastructure, e.g. `PR 1493 <https://github.com/geomstats/geomstats/pull/1493>`_.
   * Fix existing bugs, e.g. `PR 1550 <https://github.com/geomstats/geomstats/pull/1550>`_. 
   * Support for Jax backend, see `Issue 800 <https://github.com/geomstats/geomstats/issues/800>`_.

2. Documentation

   * See `documentation roadmap <https://geomstats.github.io/gsod.html>`_.

3. Learning

   * Implement geodesic PCA, see `Issue 1446 <https://github.com/geomstats/geomstats/issues/1446>`_.
   * Enhance the pool of regression algorithms on manifolds.

4. Shapes

   * Refactor the shape space of discrete open and closed curves, see `Issue 1183 <https://github.com/geomstats/geomstats/issues/1183>`_.
   * Add shape space of discrete surfaces.
   * Add shape space of deformations. 

5. Information Geometry

   * Complete manifolds of exponential families in `information_geometry  <https://github.com/geomstats/geomstats/tree/master/geomstats/information_geometry>`_.

6. Stratified Geometry

   * Complete stratified geometry of graph spaces, e.g. see `PR 1244 <https://github.com/geomstats/geomstats/pull/1244>`_.
   * Add BHV tree spaces.
