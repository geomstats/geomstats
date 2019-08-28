.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute.

The project is hosted on https://github.com/geomstats/geomstats

Code is not the only way to help the project. Reviewing pull
requests, answering questions to help others on
issues, organizing and teaching tutorials, working on the website,
improving the documentation, are all priceless contributions.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`GitHub issue tracker
<https://github.com/geomstats/geomstats/issues>`_. You are also
welcome to post feature requests or pull requests.

This contributing document is largely inspired by the Scikit-Learn project.

Ways to contribute
==================

There are many ways to contribute to geomstats, with the most common ones
being contribution of code or documentation to the project. Improving the
documentation is no less important than improving the library itself.  If you
find a typo in the documentation, or have made improvements, do not hesitate to
send an email to the mailing list or preferably submit a GitHub pull request.
Full documentation can be found under the docs/ directory.

But there are many other ways to help. In particular answering queries on the
`issue tracker <https://github.com/geomstats/geomstats/issues>`_,
investigating bugs, and :ref:`reviewing other developers' pull requests
<code_review>` are very valuable contributions that decrease the burden on the
project maintainers.

Another way to contribute is to report issues you're facing, and give a "thumbs
up" on issues that others reported and that are relevant to you.  It also helps
us if you spread the word: reference the project from your blog and articles,
link to it from your website, or simply star to say "I use it":

Submitting a bug report or a feature request
============================================

We use GitHub issues to track all bugs and feature requests; feel free to open
an issue if you have found a bug or wish to see a feature implemented.

In case you experience issues using this package, do not hesitate to submit a
ticket to the
`Bug Tracker <https://github.com/geomstats/geomstats/issues>`_. You are
also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/geomstats/geomstats/issues?q=>`_
   or `pull requests <https://github.com/geomstats/geomstats/pulls?q=>`_.

-  If you are submitting a bug report, we strongly encourage you to follow the guidelines in
   :ref:`filing_bugs`.

.. _filing_bugs:

How to make a good bug report
-----------------------------

When you submit an issue to `Github
<https://github.com/geomstats/geomstats/issues>`__, please do your best to
follow these guidelines! This will make it a lot easier to provide you with good
feedback:

- The ideal bug report contains a **short reproducible code snippet**, this way
  anyone can try to reproduce the bug easily (see `this
  <https://stackoverflow.com/help/mcve>`_ for more details). If your snippet is
  longer than around 50 lines, please link to a `gist
  <https://gist.github.com>`_ or a github repo.

- If not feasible to include a reproducible snippet, please be specific about
  what **estimators and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python, geomstats, numpy, and scipy versions**. This information
  can be found by running the following code snippet::

    >>> import geomstats
    >>> print('geomstats', geomstats.__version___)

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See `Creating and highlighting code blocks
  <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_
  for more details.

Contributing code
=================

.. note::

  To avoid duplicating work, it is highly advised that you search through the
  `issue tracker <https://github.com/geomstats/geomstats/issues>`_ and
  the `PR list <https://github.com/geomstats/geomstats/pulls>`_.
  If in doubt about duplicated work, or if you want to work on a non-trivial
  feature, it's recommended to first open an issue in
  the `issue tracker <https://github.com/geomstats/geomstats/issues>`_
  to get some feedbacks from core developers.

How to contribute
-----------------

The preferred way to contribute to geomstats is to fork the `main
repository <https://github.com/geomstats/geomstats/>`__ on GitHub,
then submit a "pull request" (PR).

In the first few steps, we explain how to locally install geomstats, and
how to set up your git repository:

1. `Create an account <https://github.com/join>`_ on
   GitHub if you do not already have one.

2. Fork the `project repository
   <https://github.com/geomstats/geomstats>`__: click on the 'Fork'
   button near the top of the page. This creates a copy of the code under your
   account on the GitHub user account. For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

3. Clone your fork of the geomstats repo from your GitHub account to your
   local disk::

       $ git clone git@github.com:YourLogin/geomstats.git
       $ cd geomstats

4. Add the ``upstream`` remote. This saves a reference to the main
   geomstats repository, which you can use to keep your repository
   synchronized with the latest changes::

    $ git remote add upstream https://github.com/geomstats/geomstats.git

You should now have a working installation of geomstats, and your git
repository properly configured. The next steps now describe the process of
modifying code and submitting a PR:

5. Synchronize your master branch with the upstream master branch::

        $ git checkout master
        $ git pull upstream master

6. Create a feature branch to hold your development changes::

        $ git checkout -b my_feature

   and start making changes. Always use a feature branch. It's good
   practice to never work on the ``master`` branch!

7. Develop the feature on your feature branch on your computer, using Git to
   do the version control. When you're done editing, add changed files using
   ``git add`` and then ``git commit``::

       $ git add modified_files
       $ git commit

   to record your changes in Git, then push the changes to your GitHub
   account with::

       $ git push -u origin my-feature

8. Follow `these
   <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   instructions to create a pull request from your fork. This will send an
   email to the committers. You may want to consider sending an email to the
   mailing list for more visibility.


It is often helpful to keep your local feature branch synchronized with the
latest changes of the main geomstats repository::

    $ git fetch upstream
    $ git merge upstream/master

Subsequently, you might need to solve the conflicts. You can refer to the
`Git documentation related to resolving merge conflict using the command
line
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_.

.. topic:: Learning git:

    The `Git documentation <https://git-scm.com/documentation>`_ and
    http://try.github.io are excellent resources to get started with git,
    and understanding all of the commands shown here.

Pull request checklist
----------------------


In order to ease the reviewing process, we recommend that your contribution
complies with the following rules. The **bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarises what your
   contribution does. This title will often become the commit message once
   merged so it should summarise your contribution for posterity. In some
   cases "Fix <ISSUE TITLE>" is enough. "Fix #<ISSUE NUMBER>" is never a
   good title.

2. **Make sure your code passes the tests**. The whole test suite can be run
   with `pytest` or `nose2`, but it is usually not recommended since it takes a long
   time. It is often enough to only run the test related to your changes:
   for example, if you changed something in
   `geomstats/spd_matrices_space.py`, running the following command will
   usually be enough:

   - `nose2 --verbose tests.test_spd_matrices_space`

   There may be other failing tests, but they will be caught by the CI so
   you don't need to run the whole test suite locally. You can read more in
   :ref:`testing_coverage`.

3. **Make sure your code is properly commented and documented**, and **make
   sure the documentation renders properly**. To build the documentation, please
   refer to our :ref:`contribute_documentation` guidelines. The CI will also
   build the docs: please refer to :ref:`generated_doc_CI`.

4. **Tests are necessary for enhancements to be
   accepted**. Bug-fixes or new features should be provided with
   `non-regression tests
   <https://en.wikipedia.org/wiki/Non-regression_testing>`_. These tests
   verify the correct behavior of the fix or feature. In this manner, further
   modifications on the code base are granted to be consistent with the
   desired behavior. In the case of bug fixes, at the time of the PR, the
   non-regression tests should fail for the code base in the master branch
   and pass for the PR code.

5. **Make sure that your PR does not add flake8 violations**. On a Unix-like
   system, you can run `make flake8-diff`. `flake8 path_to_file`, would work
   for any system, but please avoid reformatting parts of the file that your
   pull request doesn't change, as it distracts from code review.

6. Follow the `coding-guidelines`_ (see below).

7. Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should `use keywords to create link to them
   <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., ``Fixes #1234``; multiple issues/PRs are allowed as long as each
   one is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply
   related to some other issues/PRs, create a link to them without using
   the keywords (e.g., ``See also #1234``).

8. PRs should often substantiate the change, through benchmarks of
   performance and efficiency or through examples of usage. Examples also
   illustrate the features and intricacies of the library to users. Have a
   look at other examples in the `examples/
   <https://github.com/geomstats/geomstats/tree/master/examples>`_
   directory for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it to other
   methods available in geomstats.

9. New features often need to be illustrated with narrative documentation in
   the user guide, with small code snipets. If relevant, please also add
   references in the literature, with PDF links when possible.

10. The user guide should also include expected time and space complexity
    of the algorithm and scalability, e.g. "this algorithm can scale to a
    large number of samples > 100000, but does not scale in dimensionality:
    n_features is expected to be lower than 100".

You can also check our :ref:`code_review` to get an idea of what reviewers
will expect.

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).

.. note::

  The current state of the geomstats code base is not compliant with
  all of those guidelines, but we expect that enforcing those constraints
  on all new contributions will get the overall code base quality in the
  right direction.

.. note::

   For two very well documented and more detailed guides on development
   workflow, please pay a visit to the `Scipy Development Workflow
   <https://docs.scipy.org/docs/numpy/dev/gitwash/development_workflow.html>`_ -
   and the `Astropy Workflow for Developers
   <https://astropy.readthedocs.io/en/latest/development/workflow/development_workflow.html>`_
   sections.

Continuous Integration (CI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* CircleCI is used to build the docs for viewing and for linting with flake8.

Please note that if one of the following markers appear in the latest commit
message, the following actions are taken.

    ====================== ===================
    Commit Message Marker  Action Taken by CI
    ---------------------- -------------------
    [scipy-dev]            Add a Travis build with our dependencies (numpy, scipy, etc ...) development builds
    [ci skip]              CI is skipped completely
    [doc skip]             Docs are not built
    [doc quick]            Docs built, but excludes example gallery plots
    [doc build]            Docs built including example gallery plots
    ====================== ===================

Stalled pull requests
^^^^^^^^^^^^^^^^^^^^^

As contributing a feature can be a lengthy process, some
pull requests appear inactive but unfinished. In such a case, taking
them over is a great service for the project.

A good etiquette to take over is:

* **Determine if a PR is stalled**

  * A pull request may have the label "stalled" or "help wanted" if we
    have already identified it as a candidate for other contributors.

  * To decide whether an inactive PR is stalled, ask the contributor if
    she/he plans to continue working on the PR in the near future.
    Failure to respond within 2 weeks with an activity that moves the PR
    forward suggests that the PR is stalled and will result in tagging
    that PR with "help wanted".

    Note that if a PR has received earlier comments on the contribution
    that have had no reply in a month, it is safe to assume that the PR
    is stalled and to shorten the wait time to one day.

    After a sprint, follow-up for un-merged PRs opened during sprint will
    be communicated to participants at the sprint, and those PRs will be
    tagged "sprint". PRs tagged with "sprint" can be reassigned or
    declared stalled by sprint leaders.

* **Taking over a stalled PR**: To take over a PR, it is important to
  comment on the stalled PR that you are taking over and to link from the
  new PR to the old one. The new PR should be created by pulling from the
  old one.

.. _new_contributors:

Issues for New Contributors
---------------------------

New contributors should look for the following tags when looking for issues.  We
strongly recommend that new contributors tackle "easy" issues first: this helps
the contributor become familiar with the contribution workflow, and for the core
devs to become acquainted with the contributor; besides which, we frequently
underestimate how easy an issue is to solve!

.. topic:: good first issue tag

    A great way to start contributing to geomstats is to pick an item from
    the list of `good first issues
    <https://github.com/geomstats/geomstats/labels/good%20first%20issue>`_
    in the issue tracker. Resolving these issues allow you to start contributing
    to the project without much prior knowledge. If you have already contributed
    to geomstats, you should look at Easy issues instead.

.. topic:: starter project tag

    If you have already contributed to geomstats, another great way to contribute
    to geomstats is to pick an item from the list of `starter project
    <https://github.com/geomstats/geomstats/labels/starter%project>`_ in the issue
    tracker. Your assistance in this area will be greatly appreciated by the
    more experienced developers as it helps free up their time to concentrate on
    other issues.

.. topic:: help wanted tag

    We often use the help wanted tag to mark issues regardless of difficulty. Additionally,
    we use the help wanted tag to mark Pull Requests which have been abandoned
    by their original contributor and are available for someone to pick up where the original
    contributor left off. The list of issues with the help wanted tag can be found
    `here <https://github.com/geomstats/geomstats/labels/help%20wanted>`__ .

    Note that not all issues which need contributors will have this tag.

.. _contribute_documentation:

Documentation
=============

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc. reStructuredText
documents live in the source code repository under the ``docs/`` directory.

You can edit the documentation using any text editor, and then generate the
HTML output by typing ``make`` from the ``docs/`` directory. Alternatively,
``make html`` may be used to generate the documentation **with** the example
gallery (which takes quite some time). The resulting HTML files will be
placed in ``_build/html/stable`` and are viewable in a web browser.


Building the documentation
--------------------------

First, make sure you have :ref:`properly installed <install_bleeding_edge>`
the development version.

Building the documentation requires installing some additional packages::

    pip install sphinx sphinx-gallery numpydoc matplotlib Pillow pandas scikit-image

To build the documentation, you need to be in the ``docs`` folder::

    cd docs

In the vast majority of cases, you only need to generate the full web site::

    make html


Guidelines for writing documentation
------------------------------------

It is important to keep a good compromise between mathematical and algorithmic
details, and give intuition to the reader on what the algorithm does.

Basically, to elaborate on the above, it is best to always
start with a small paragraph with a hand-waving explanation of what the
method does to the data. Then, it is very helpful to point out why the feature is
useful and when it should be used - the latter also including "big O"
(:math:`O\left(g\left(n\right)\right)`) complexities of the algorithm, as opposed
to just *rules of thumb*, as the latter can be very machine-dependent. If those
complexities are not available, then rules of thumb may be provided instead.

Secondly, a generated figure from an example (as mentioned in the previous
paragraph) should then be included to further provide some intuition.

Next, one or two small code examples to show its use can be added.

Next, any math and equations, followed by references,
can be added to further the documentation. Not starting the
documentation with the maths makes it more friendly towards
users that are just interested in what the feature will do, as
opposed to how it works "under the hood".

Finally, follow the formatting rules below to make it consistently good:

* Add "See also" in docstrings for related classes/functions.

* "See also" in docstrings should be one line per reference,
  with a colon and an explanation.

* When documenting the parameters and attributes, here is a list of some
  well-formatted examples::

    n_clusters : int, default=3
        The number of clusters detected by the algorithm.

    some_param : {'hello', 'goodbye'}, bool or int, default=True
        The parameter description goes here, which can be either a string
        literal (either `hello` or `goodbye`), a bool, or an int. The default
        value is True.

    array_parameter : {array-like, sparse matrix, dataframe} of shape (n_samples, n_features) or (n_samples,)
        This parameter accepts data in either of the mentioned forms, with one
        of the mentioned shapes. The default value is
        `np.ones(shape=(n_samples,))`.

In general have the following in mind:

    1. Use Python basic types. (``bool`` instead of ``boolean``)
    2. Use parenthesis for defining shapes: ``array-like of shape (n_samples,)``
       or ``array-like of shape (n_samples, n_features)``
    3. For strings with multiple options, use brackets:
       ``input: {'log', 'squared', 'multinomial'}``
    4. 1D or 2D data can be a subset of
       ``{array-like, ndarray, sparse matrix, dataframe}``. Note that ``array-like``
       can also be a ``list``, while ``ndarray`` is explicitly only a ``numpy.ndarray``.

* When editing reStructuredText (``.rst``) files, try to keep line length under
  80 characters when possible (exceptions include links and tables).

.. _testing_coverage:

Testing and improving test coverage
===================================

High-quality `unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_
is a corner-stone of the geomstats development process. For this
purpose, we use the `nose2`
package. The tests are functions appropriately named, located in `tests`
subdirectories, that check the validity of the algorithms and the
different options of the code.

We expect code coverage of new features to be at least around 90%.

.. _coding-guidelines:

Coding guidelines
=================

The following are some guidelines on how new code should be written. Of
course, there are special cases and there will be exceptions to these
rules. However, following these rules when submitting new code makes
the review easier so new code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The
geomstats project tries to closely follow the official Python guidelines
detailed in `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ that
detail how code should be formatted and indented. Please read it and
follow it.

In addition, we add the following guidelines:

* Use underscores to separate words in non class names: ``n_samples``
  rather than ``nsamples``.

* Avoid multiple statements on one line. Prefer a line return after
  a control flow statement (``if``/``for``).

* **Please don't use** ``import *`` **in any case**. It is considered harmful
  by the `official Python recommendations
  <https://docs.python.org/2/howto/doanddont.html#from-module-import>`_.
  It makes the code harder to read as the origin of symbols is no
  longer explicitly referenced, but most important, it prevents
  using a static analysis tool like `pyflakes
  <https://divmod.readthedocs.io/en/latest/products/pyflakes.html>`_ to automatically
  find bugs in geomstats.

* Use the `numpy docstring standard
  <https://github.com/numpy/numpy/blob/master/docs/HOWTO_DOCUMENT.rst.txt>`_
  in all your docstrings.


A good example of code that we like can be found `here
<https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01>`_.


.. currentmodule:: geomstats

.. _code_review:

Code Review Guidelines
======================
Reviewing code contributed to the project as PRs is a crucial component of
geomstats development. We encourage anyone to start reviewing code of other
developers. The code review process is often highly educational for everybody
involved. This is particularly appropriate if it is a feature you would like to
use, and so can respond critically about whether the PR meets your needs. While
each pull request needs to be signed off by two core developers, you can speed
up this process by providing your feedback.

Here are a few important aspects that need to be covered in any code review,
from high-level questions to a more detailed check-list.

- Do we want this in the library? Is it likely to be used? Do you, as
  a geomstats user, like the change and intend to use it? Is it in
  the scope of geomstats? Will the cost of maintaining a new
  feature be worth its benefits?

- Is the code consistent with the API of geomstats? Are public
  functions/classes/parameters well named and intuitively designed?

- Are all public functions/classes and their parameters, return types, and
  stored attributes named according to geomstats conventions and documented clearly?

- Is any new functionality described in the user-guide and illustrated with examples?

- Is every public function/class tested? Are a reasonable set of
  parameters, their values, value types, and combinations tested? Do
  the tests validate that the code is correct, i.e. doing what the
  documentation says it does? If the change is a bug-fix, is a
  non-regression test included? Look at `this
  <https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing>`__
  to get started with testing in Python.

- Do the tests pass in the continuous integration build? If
  appropriate, help the contributor understand why tests failed.

- Do the tests cover every line of code (see the coverage report in the build
  log)? If not, are the lines missing coverage good exceptions?

- Is the code easy to read and low on redundancy? Should variable names be
  improved for clarity or consistency? Should comments be added? Should comments
  be removed as unhelpful or extraneous?

- Could the code easily be rewritten to run much more efficiently for
  relevant settings?

- Will the new code add any dependencies on other libraries? (this is
  unlikely to be accepted)

- Does the documentation render properly (see the
  :ref:`contribute_documentation` section for more details), and are the plots
  instructive?

:ref:`saved_replies` includes some frequent comments that reviewers may make.
