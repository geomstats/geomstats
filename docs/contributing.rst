.. _contributing:

============
Contributing
============

Geomstats is a community effort, and everyone is welcome to
contribute.

The project is hosted on https://github.com/geomstats/geomstats.

Code is not the only way to help the project:

- reviewing pull requests,
- submitting an issue or a bug report,
- submitting a thumb up on issues that are important for you,
- answering questions to help others on issues,
- requesting new features,
- improving the coding style in the codebase,
- improving the coding style guidelines,
- organizing and teaching tutorials, organizing hackathons,
- working on the website and improving the documentation,
- reference the project in your articles,
- or simply star it to say "I use it",

are all priceless contributions.

If you have questions, do not hesitate to send an email to the mailing
list hi@geomstats.ai.

In this contributing document, we give details on submitting issues,
contributing code, writing documentation, following code style guidelines,
and reviewing code. This document is largely inspired by the
Scikit-Learn project.

Submitting a Bug Report or Feature Request
------------------------------------------

We use GitHub issues to track all bugs and feature requests; feel free to `open
an issue <https://github.com/geomstats/geomstats/issues>`_ if you have found a
bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/geomstats/geomstats/issues?q=>`_
   or `pull requests <https://github.com/geomstats/geomstats/pulls?q=>`_.

-  If you are submitting a bug report, we strongly encourage you to
   follow the guidelines in :ref:`filing_bugs`.

.. _filing_bugs:

How to Write a Good Bug Report
==============================

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

Contributing: Code Workflow
---------------------------

  To avoid duplicating work, it is highly advised that you search through the
  `issue tracker <https://github.com/geomstats/geomstats/issues>`_ and
  the `PR list <https://github.com/geomstats/geomstats/pulls>`_.
  If in doubt about duplicated work, or if you want to work on a non-trivial
  feature, it's recommended to first open an issue in
  the `issue tracker <https://github.com/geomstats/geomstats/issues>`_
  to get some feedbacks from core developers.


Contributing: Adding a new geometry or manifold
-----------------------------------------------

  Get used to the code design of geomstats' classes by having a look at the `template manifold <https://github.com/geomstats/geomstats/tree/master/geomstats/geometry/_my_manifold.py>`__ and `template unit-tests <https://github.com/geomstats/geomstats/blob/master/tests/tests_geomstats/test__my_manifold.py>`__ implementations.

Contributing: GitHub Workflow
=============================

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
       $ git commit -m "Add my feature"

   to record your changes in Git. Your commit message should respect the `good
   commit messages guidelines <https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project>`_.
   Then push the changes to your GitHub account with::

       $ git push -u origin my-feature

8. Follow `these
   <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   instructions to create a pull request from your fork. This will send an
   email to the committers. You may want to consider sending an email to the
   mailing list hi@geomstats.ai for more visibility.

It is often helpful to keep your local feature branch synchronized with the
latest changes of the main geomstats repository. If there are only a few new 
commits in the master branch, use::

    $ git fetch upstream
    $ git rebase upstream/master

Subsequently, you might need to solve potential conflicts.
If it's been a while since you've last updated your branch, it might be easier
to merge the master branch into yours::

    $ git fetch upstream
    $ git merge upstream/master

Refer to the
`Git documentation related to resolving merge conflict using the command
line
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_.
The `Git documentation <https://git-scm.com/documentation>`_ and
http://try.github.io are excellent resources to get started with git,
and understanding all of the commands shown here.

Pull Request Checklist
======================

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules. The **bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarises what your
   contribution does. This title will often become the commit message once
   merged so it should summarise your contribution for posterity. In some
   cases "Fix <ISSUE TITLE>" is enough. "Fix #<ISSUE NUMBER>" is never a
   good title.

2. **Make sure that your code is** `vectorized
   <https://www.geeksforgeeks.org/vectorization-in-python/>`_. For vectorized matrix operations we recommend using the
   methods of the  `Matrices <https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/matrices.py>`_
   class instead of lower level backend functions, as they are automatically vectorized.

3. **Submit your code with associated unit tests**. High-quality
   `unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_
   is a corner-stone of the geomstats development process.
   The tests are functions appropriately named, located in the `tests`
   subdirectory, that check the validity of the algorithms and the
   different options of the code. For each function `my_fun` that you
   implement in a given `my_module.py`, you should add the corresponding test
   function `test_my_fun` in the file `test_my_module.py`. We expect code
   coverage of new features to be at least 90%, which is automatically verified
   by the `codecov` software when you submit a PR. You should also add
   `test_my_fun_vect` tests to ensure that your code is vectorized.

4. **Make sure your code passes all unit tests**. First,
   run the tests related to your changes. For example, if you changed
   something in `geomstats/spd_matrices_space.py`:

   `nose2 --verbose tests.test_spd_matrices_space`

   and then run the tests of the whole codebase to check that your feature is
   not breaking any of them:

   `nose2`

   This way, further modifications on the code base are granted
   to be consistent with the desired behavior. Merging your PR should not break
   any test in any backend (numpy, tensorflow or pytorch).

5. **Make sure that your PR follows Python international style guidelines**,
   `PEP8 <https://www.python.org/dev/peps/pep-0008>`_, which you should read.
   The `flake8` package automatically checks for style violations when you
   submit your PR. We recommend installing flake8 with its plugins on your
   machine by running

   `pip3 install -r dev-requirements.txt`

   Then you can run

   `flake8 geomstats tests examples`

   To prevent adding commits which fail to adhere to the PEP8 guidelines, we
   include a `pre-commit <https://pre-commit.com/>` config, which immediately
   invokes flake8 on all files staged for commit when running `git commit`. To
   enable the hook, simply run `pre-commit install` after installing
   `pre-commit` either manually via `pip` or as part of `dev-requirements.txt`.

   Please avoid reformatting parts of the file that your pull request doesn't
   change, as it distracts during code reviews.

6. **Make sure that your PR follows geomstats coding style and API**,
   see our `coding-guidelines`_ below. Ensuring style consistency throughout
   geomstats allows using tools to automatically parse the codebase,
   for example searching all instances where a given function is used,
   or use automatic find-and-replace during code's refactorizations. It
   also speeds up the code review and acceptance of PR, as the maintainers
   do not spend time getting used to new conventions and coding preferences.

7. **Make sure your code is properly documented**, and **make
   sure the documentation renders properly**. To build the documentation, please
   see our :ref:`contribute_documentation` guidelines. The plugin
   flake8-docstrings automatically checks that your the documentation follows
   our guidelines when you submit a PR.

8. Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should `use keywords to create link to them
   <https://github.com/blog/1506-closing-issues-via-pull-requests/>`_
   (e.g., ``Fixes #1234``; multiple issues/PRs are allowed as long as each
   one is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply
   related to some other issues/PRs, create a link to them without using
   the keywords (e.g., ``See Also #1234``).

9. PRs should often substantiate the change, through benchmarks of
   performance and efficiency or through examples of usage. Examples also
   illustrate the features and intricacies of the library to users. Have a
   look at other examples in the `examples/
   <https://github.com/geomstats/geomstats/tree/master/examples>`_
   directory for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it to other
   methods available in geomstats.

10. The user guide should also include expected time and space complexity
    of the algorithm and scalability, e.g. "this algorithm can scale to a
    large number of samples > 100000, but does not scale in dimensionality:
    n_features is expected to be lower than 100".

11. **Each PR needs to be accepted by a core developer** before
    being merged.

You can also check our :ref:`code_review` to get an idea of what reviewers
will expect.

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list hi@geomstats.ai or on the GitHub issue).

.. note::

  The current state of the geomstats code base is not compliant with
  all of those guidelines, but we expect that enforcing those constraints
  on all new contributions will get the overall code base quality in the
  right direction.

Stalled Pull Requests
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

* **Taking over a stalled PR**: To take over a PR, it is important to
  comment on the stalled PR that you are taking over and to link from the
  new PR to the old one. The new PR should be created by pulling from the
  old one.

.. _new_contributors:

Issues for New Contributors
===========================

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
-------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc. reStructuredText
documents live in the source code repository under the ``docs/`` directory.

Building the Documentation
==========================

Building the documentation requires installing sphinx::

   pip3 install sphinx

To build the documentation, you need to be in the main ``geomstats`` folder. You can do this with::

   sphinx-build docs/ docs/html

Writing Docstrings
==================

Intro to Docstrings
^^^^^^^^^^^^^^^^^^^

A docstring is a well-formatted description of your function/class/module which includes
its purpose, usage, and other information.

There are different markdown languages/formats used for docstrings in Python. The most common
three are reStructuredText, numpy, and google docstring styles. For geomstats, we are
using the numpy docstring standard.
When writing up your docstrings, please review the `NumPy docstring guidge <https://numpydoc.readthedocs.io/en/latest/format.html>`_
to understand the role and syntax of each section. Following this syntax is important not only for readability,
it is also required for automated parsing for inclusion into our generated API Reference.

You can look at these for any object by printing out the ``__doc__`` attribute.
Try this out with the np.array class and the np.mean function to see good examples::

    >>> import numpy as np
    >>> print(np.mean.__doc__)

The Anatomy of a Docstring
^^^^^^^^^^^^^^^^^^^^^^^^^^

These are some of the most common elements for functions (and ones we’d like you to add where appropriate):

1. Summary - a one-line (here <79 char) description of the object

   a. Begins immediately after the first """ with a capital letter, ends with a period

   b. If describing a function, use a verb with the imperative mood (e.g. **Compute** vs Computes)

   c. Use a verb which is as specific as possible, but default to Compute when uncertain (as opposed to Calculate or Evaluate, for example)

2. Description - a more informative multi-line description of the function

   a. Separated from the summary line by a blank line

   b. Begins with a capital letter and ends with period

3. Parameters - a formatted list of arguments with type information and description

   a. On the first line, state the parameter name, type, and shape when appropriate. The parameter name should be separated from the rest of the line by a ``:`` (with a space on either side). If a parameter is optional, write ``Optional, default: default_value.`` as a separate line in the description.
   b. On the next line, indent and write a summary of the parameter beginning with a capital letter and ending with a period.

   c. See :ref:`docstring_examples` below

4. Returns (esp. for functions) - a formatted list of returned objects type information and description

   a. The syntax here is the same as in the parameters section above.

   b. See :ref:`docstring_examples` below

If documenting a class, you would also want to include an Attributes section.
There are many other optional sections you can include which are very helpful.
For example: Raises, See Also, Notes, Examples, References, etc.

N.B. Within Notes, you can
	- include LaTex code
	- cite references in text using ids placed in References

.. _docstring_examples:

Docstring Examples
^^^^^^^^^^^^^^^^^^
Here's a generic docstring template::

   def my_method(self, my_param_1, my_param_2='vector'):
      """Write a one-line summary for the method.

      Write a description of the method, including "big O"
      (:math:`O\left(g\left(n\right)\right)`) complexities.

      Parameters
      ----------
      my_param_1 : array-like, shape=[..., dim]
         Write a short description of parameter my_param_1.
      my_param_2 : str, {'vector', 'matrix'}
         Write a short description of parameter my_param_2.
         Optional, default: 'vector'.

      Returns
      -------
      my_result : array-like, shape=[..., dim, dim]
         Write a short description of the result returned by the method.

      Notes
      -----
      If relevant, provide equations with (:math:)
      describing computations performed in the method.

      Example
      -------
      Provide code snippets showing how the method is used.
      You can link to scripts of the examples/ directory.

      Reference
      ---------
      If relevant, provide a reference with associated pdf or
      wikipedia page.
      """

And here's a filled-in example from the Scikit-Learn project, modified to our syntax::

   def fit_predict(self, X, y=None, sample_weight=None):
      """Compute cluster centers and predict cluster index for each sample.

      Convenience method; equivalent to calling fit(X) followed by
      predict(X).

      Parameters
      ----------
      X : {array-like, sparse_matrix} of shape=[..., n_features]
         New data to transform.
      y : Ignored
         Not used, present here for API consistency by convention.
      sample_weight : array-like, shape [...,], optional
         The weights for each observation in X. If None, all observations
         are assigned equal weight (default: None).

      Returns
      -------
      labels : array, shape=[...,]
         Index of the cluster each sample belongs to.
      """
      return self.fit(X, sample_weight=sample_weight).labels_

In general, have the following in mind:

   1. Use built-in Python types. (``bool`` instead of ``boolean``)

   2. Use ``[`` for defining shapes: ``array-like, shape=[..., dim]``

   3. If a shape can vary, use a list-like notation:
      ``array-like, shape=[dimension[:axis], n, dimension[axis:]]``

   4. For strings with multiple options, use brackets:
      ``input: str, {'log', 'squared', 'multinomial'}``

   5. 1D or 2D data can be a subset of
      ``{array-like, ndarray, sparse matrix, dataframe}``. Note that
      ``array-like`` can also be a ``list``, while ``ndarray`` is explicitly
      only a ``numpy.ndarray``.

   6. Add "See Also" in docstrings for related classes/functions.
      "See Also" in docstrings should be one line per reference,
      with a colon and an explanation.

For Class and Module Examples see the `scikit-learn _weight_boosting.py module
<https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/ensemble/_weight_boosting.py#L285>`_.
The class AdaBoost has a great example using the elements we’ve discussed here.
Of course, these examples are rather verbose, but they’re good for
understanding the components.

When editing reStructuredText (``.rst``) files, try to keep line length under
80 characters (exceptions include links and tables).

.. _coding-guidelines:

Coding Style Guidelines
-----------------------

The following are some guidelines on how new code should be written. Of
course, there are special cases and there will be exceptions to these
rules. However, following these rules when submitting new code makes
the review easier so new code can be integrated in less time. Uniformly
formatted code makes it easier to share code ownership.

In addition to the PEP8 standards, geomstats follows the following
guidelines:

1. Use underscores to separate words in non class names: ``n_samples``
   rather than ``nsamples``.

2. Avoid single-character variable names. This prevents using automatic
   tools to find-and-replace code, as searching for ``x`` in geomstats
   will return the whole database. At least 3 characters are advised
   for a variable name.

3. Use understandable function and variable names. The naming should help
   the maintainers reading faster through your code. Thus, ``my_array``,
   ``aaa``, ``result``, ``res`` are generally bad variable names,
   whereas ``rotation_vec`` or ``symmetric_mat`` read well.

4. Avoid comment in the code, the documentation goes in the docstrings.
   This allows the explanations to be included in the documentation
   generated automatically on the website. Furthermore, forbidding comments
   forces us to write clean code, and clean docstrings.

5. Follow geomstats' API. For example, points on manifolds are denoted
   ``point``, tangent vectors ``tangent_vec``, matrices ``mat``, exponential
   ``exp`` and logarithms ``log``.

6. Avoid multiple statements on one line. Divide complex computations on several
   lines. Prefer a line return after a control flow statement (``if``/``for``).

7. **Don't use** ``import *`` **in any case**. It is considered harmful
   by the `official Python recommendations
   <https://docs.python.org/2/howto/doanddont.html#from-module-import>`_.
   It makes the code harder to read as the origin of symbols is no
   longer explicitly referenced, but most important, it prevents
   using a static analysis tool like `pyflakes
   <https://divmod.readthedocs.io/en/latest/products/pyflakes.html>`_ to
   automatically find bugs in geomstats.

8. Avoid the use of ``import ... as`` and of ``from ... import foo, bar``,
   i.e. do not rename modules or modules' functions, because you would create
   objects living in several namespaces which creates confusion, see
   `<https://docs.python.org/2/howto/doanddont.html#language-constructs-you-should-not-use>`_.
   Keeping the original namespace ensures naming consistency in the codebase
   and speeds up the code reviews: co-developpers and maintainers do not have
   to check if you are using the original module's method or if you have
   overwritten it.

9. Use single quotes ' and not double quotes " for strings.

10. If you need several lines for a function call, use the syntax
::
   my_function_with_a_very_long_name(
      my_param_1=value_1, my_param_2=value_2)

and not
::
   my_function_with_a_very_long_name(my_param_1=value_1,
                                     my_param_2=value_2)

as the indentation will break and raise a flake8 error if the name
of the function is changed.

These guidelines can be revised and modified at any time, the only constraint
is that they should remain consistent through the codebase. To change geomstats
style guidelines, submit a PR to this contributing file, together with the
corresponding changes in the codebase.

.. _code_review:

Code Review Guidelines
----------------------

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
  stored attributes named according to geomstats conventions and documented
  clearly?

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
  improved for clarity or consistency?

- Could the code easily be rewritten to run much more efficiently for
  relevant settings?

- Will the new code add any dependencies on other libraries? (this is
  unlikely to be accepted)

- Does the documentation render properly (see the
  :ref:`contribute_documentation` section for more details), and are the plots
  instructive?

- Upon merging, use the ``Rebase and Merge`` option to keep git history clean.
