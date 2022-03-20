.. _gsod:

===================================
Google Season of Docs: Project Page
===================================

Design and Organize Scattered Documentation - Geomstats

About the organization
======================

Geomstats is an open-source Python package for computations and statistics on nonlinear manifolds, such as hyperspheres, hyperbolic spaces, Lie groups of transformations, among others. As such, the package supports research within the fast growing field of Geometric (Deep) Learning.

In this context, Geomstats provides code to fulfill four objectives:

- provide educational support to learn "hands-on" differential geometry and geometric statistics and learning, through its examples and visualizations.
- foster research in differential geometry and geometric statistics by providing operations on manifolds to gain intuition on results of a research paper;
- democratize the use of geometric statistics by implementing user-friendly geometric learning algorithms using Scikit-Learn API; and
- provide a platform to share learning algorithms on manifolds.

The `source code <https://github.com/geomstats/geomstats>`_ is freely available on GitHub. Geomstats is organized into two main modules:
`geometry` and `learning`. The module `geometry` implements concepts in differential geometry. The module `learning` implements statistics and learning algorithms for data on manifolds. The code is object-oriented and classes inherit from scikit-learn's base classes and mixins.


Geomstats has already found numerous applications, for example in the biomedical fields for machine learning applied to biological shape analysis. Typical usecases can be found within the `notebooks folder <https://github.com/geomstats/geomstats/blob/master/notebooks/>`_.

About the project
=================

The problem
-----------

Current Geomstats documentation has two main problems:
- the documentation website lacks a professional design,
- the documentation is scattered across docstrings, example python scripts, how-to jupyter notebooks, tutorials, arxiv review papers.

As a result, potential new contributors often fail to understand the implementation of the library and renounce in participating to its development.

We aim to adopt the documentation system of divio.


The project’s scope
-------------------

The Geomstats project will:

- Propose and implement a new design for Geomstats documentation website.
- Audit existing Geomstats documentation across Python scripts, jupyter notebooks, docstrings and pdf documents.
- Organize existing documentation within the website, using divio documentation principles.
- Update README.rst and CONTRIBUTING.rst with the aim to better onboard new contributors; Add a checklist for new contributors' first Pull Requests.
- Add missing references and mathematical formulas to docstrings.
- Incorporate feedback from documentation testers and Geomstats community.


Work that is out-of-scope for this project:
- This project will not create any explanation, tutorials or how-to guides.

We estimate that this work will take 6 months to complete.

Measuring project’s success
---------------------------

Geomstats receives an average of 50 pull requests a quarter to add a new feature, propose a bug fix or work on the software coding infrastructure. A third of these pull requests are from new contributors. However:
- new contributors' pull requests are rarely merged, as they do not obide Geomstats coding guidelines, and stall as a result,
- even if the pull request is merged, the new contributor will not submit another one.
We believe that the improved documentation will result in more pull requests from new contributors that will be merged, i.e. less stalled pull requests, and that the enhanced contribution experience will motivate new contributors to submit a second pull request.

We will track the following metrics:
- (a) monthly: number of pull requests from new contributors,
- (b) monthly: percentage of pull requests from new contributors that are merged (within two weeks),
- (c) monthly: number of contributors with >2 pull requests.
- (d) standard metrics (number of views, downloads, web traffic to documentation website, time on page), monthly after the documentation is published.


We would consider the project successful if, after publication of the new documentation at least three of the following hold:

- The number of pull requests by new contributors increases by 15%.
- The percentage of pull requests by new contributors that are merged within two weeks increases by 15%.
- The number of new contributors with >2 merged pull requests increases by 10% (beginning the quarter after the documentation is published).
- The standard metrics increase by 10% on average.
- The number of GitHub forks and stars increases by 10%.
