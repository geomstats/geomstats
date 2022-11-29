.. _gsod:


=================================
Google Season of Docs: Case Study
=================================


Design and Organize Scattered Documentation in Geomstats
========================================================

* Organization: Geomstats.

The Season of Docs page is at the botton of this page.

* Organization Description: 

Geomstats is an open-source Python package for computations and statistics on nonlinear manifolds, such as hyperspheres, hyperbolic spaces, Lie groups of transformations, among others. As such, the package supports research within the fast growing field of Geometric (Deep) Learning.

The `source code <https://github.com/geomstats/geomstats>`_ is freely available on GitHub. Geomstats is organized into two main modules:
`geometry` and `learning`. The module `geometry` implements concepts in differential geometry. The module `learning` implements statistics and learning algorithms for data on manifolds. The code is object-oriented and classes inherit from scikit-learn's base classes and mixins.

Geomstats has already found numerous applications, for example in the biomedical fields for machine learning applied to biological shape analysis. Typical usecases can be found within the `notebooks folder <https://github.com/geomstats/geomstats/blob/master/notebooks/>`_.


Problem Statement
=================

Current Geomstats documentation has two main problems:
- the documentation website lacks a professional design,
- the documentation is scattered across docstrings, example python scripts, how-to jupyter notebooks, tutorials, arxiv review papers.

As a result, potential new contributors often fail to understand the implementation of the library and renounce in participating to its development.

The goal of this project is to adopt the documentation system of divio, to design and reorganize our scattered documentation.

Proposal Abstract
=================

Our original organization proposal stated the following goals towards solving the problem stated above:

- Propose and implement a new design for Geomstats documentation website.
- Audit existing Geomstats documentation across Python scripts, jupyter notebooks, docstrings and pdf documents.
- Organize existing documentation within the website, using divio documentation principles.
- Update README.rst and CONTRIBUTING.rst with the aim to better onboard new contributors; Add a checklist for new contributors' first Pull Requests.
- Add missing references and mathematical formulas to docstrings.
- Incorporate feedback from documentation testers and Geomstats community.

The original proposal page can be found at the bottom of this webpage.

Project Description
===================

There were no obstacles to coming up with our Season of Docs proposal. We were discussing our lack of resources in order to improve our documentation for a long time. We were aware of its issues by our frequent discussions with contributors and users. The proposal organically emerged by summarizing all feedback and comments made during these discussions.


Budget
======

The proposed budget is included in the original proposal which is found at the botton of this page.

We found it hard to estimate the amount of work required for each of the tasks, as this is dependent on the qualifications of the candidate. We decided to give general objectives to achieve, as opposed to explicit number of lines or web pages to write.

They were no unexpected expenses. Our estimates were correct. We have other resources on Geomstats that we were able to use, e.g. the salaries of engineers working on the repository.


Participants
============

Jonahh
Who worked on this project (use usernames if requested by participants)? How did you find and hire your technical writer? How did you find other volunteers or paid participants? What roles did they have? Did anyone drop out? What did you learn about recruiting, communication, and project management?

Timeline
========
*_Give a short overview of the timeline of your project (indicate estimated end date or intermediate milestones if project is ongoing).Did the original timeline need adjustment?_*

Results
========
*_What was created, updated, or otherwise changed? Include links to published documentation if available. Were there any deliverables in the proposal that did not get created? List those as well. Did this project result in any new or updated processes or procedures in your organization?_*

Metrics
========
*_What metrics did you choose to measure the success of the project? Were you able to collect those metrics? Did the metrics correlate well or poorly with the behaviors or outcomes you wanted for the project? Did your metrics change since your proposal? Did you add or remove any metrics? How often do you intend to collect metrics going forward?_*

Analysis
========

We believe that this project was successful because the documentation seems more accessible. However, we will need to evaluate our metrics (given in the original proposal, see below) over the course of the next year and see the informal feedback gathered through our discussions with contributors and users.

Summary
=======

The Google Season of Docs was a great experience. It was very interesting to have a technical writer originally external to the project delve into the documentation. It highlights specifically well the problems that can exist within our current documentation website. 

In the future, we would probably think more ahead in terms of the time zones of the different participants. As our participants were spanning several continents, it was sometimes complicated to have continued discussions.


*The above case study was conducted based on the project originally proposed, which is given below.*

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

We estimate that this work will take 6 months to complete. We are in contact with interested technical writers, some having already merged documentation pull requests to the repository.

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


Proposed Budget
===============

.. list-table::
   :header-rows: 1

   * - **Budget Item**
     - **Amount**
     - **Running Total**
     - **Notes**
   * - Technical writer
     - 10,000.00
     - 10,000.00
     - One technical writer for 6 months.
   * - Volunteer stipends
     - 500.00
     - 13,000.00
     - Six volunteers stipends.
   * - Project T-shirts and stickers
     - 150.00
     - 14,500.00
     - Ten project T-shirts, ten stickers.
   * - **Total**
     -
     - **14,500.00**
     -

Additional information
======================

Nina Miolane, principal administrator, has been working with a master student to create the original Geomstats documentation website. She has worked with two engineers to update the documentation website in 2021.

Nina Miolane has been a mentor for `Outreachy <https://www.outreachy.org/>`_ and is a mentor + member of the steering committee of the MIT's `Summer Geometry Initiative <https://sgi.mit.edu/>`_, two programs that are similar to Google Summer of Code and Google Season of Docs. Nina Miolane has co-organized `hackathons <https://github.com/geomstats/geomstats#contributing>`_ , `coding challenges <https://github.com/geomstats/challenge-iclr-2021>`_, and `international workshops <https://www.ninamiolane.com/pagecv>`_.

Important links for technical writers
=====================================

- Find details about being a Google SoD's writer `here <https://developers.google.com/season-of-docs/docs/tech-writer-guide>`_.
- Feel free to register your interest on the `SoD 2022 github repository <https://github.com/google/season-of-docs/tree/main/2022-participants>`_.
- Submit your statement of interest via email to nmiolane@gmail.com before May 1st, at 18:00 UTC.
- Questions? Join our community through its `slack workspace <https://join.slack.com/t/geomstats/shared_invite/zt-15t8k0q7r-Duxt6j6aA3rDOIms2Vd~RQ>`_!
