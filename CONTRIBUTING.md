# Contributing to Geomstats

Welcome to the geomstats repository!
We're excited you're here and want to contribute.

## Practical Guide to Submitting your Contribution

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below,
please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in.
Here are some [instructions][link_signupinstructions].


## Joining the Conversation

`geomstats` is maintained by a growing group of enthusiastic developers!
Most of our discussions take place on [issues][link_issues].


## Contributing through GitHub

[git][link_git] is a really useful tool for version control.
[GitHub][link_github] sits on top of git and supports collaborative and distributed working.

If you're not yet familiar with `git`, there are lots of great resources to help you *git* started!
Some of our favorites include the [git Handbook][link_handbook] and
the [Software Carpentry introduction to git][link_swc_intro].

On GitHub, you'll use [Markdown][markdown] to chat in issues and pull requests.
You can think of Markdown as a few little symbols around your text that will allow GitHub
to render the text with a little bit of formatting.
For example you could write words as bold (`**bold**`), or in italics (`*italics*`),
or as a [link][rick_roll] (`[link](https://https://youtu.be/dQw4w9WgXcQ)`) to another webpage.

GitHub has a really helpful page for getting started with
[writing and formatting Markdown on GitHub][writing_formatting_github].


## Understanding Issues

Every project on GitHub uses [issues][link_issues] slightly differently.

The following outlines how ``geomstats`` developers think about these tools.

* **Issues** are individual pieces of work that need to be completed to move the project forwards.
A general guideline: if you find yourself tempted to write a great big issue that
is difficult to describe as one unit of work, please consider splitting it into two or more issues.

* Issues are assigned [labels](#issue-labels) which explain how they relate to the overall project's
goals and immediate next steps.


## Making a Change

We appreciate all contributions to ``geomstats``,
but those accepted fastest will follow a workflow similar to the following:

**1. Comment on an existing issue or open a new issue referencing your addition.**

This allows other members of the ``geomstats`` development team to confirm that you aren't
overlapping with work that's currently underway and that everyone is on the same page
with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front
is so useful to everyone involved.

**2. [Fork][link_fork] the [geomstats repository][link_geomstats] to your profile.**

This is now your own unique copy of ``geomstats``.
Changes here won't effect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date][link_updateupstreamwiki] with the master repository.

**3. Make the changes you've discussed, following the [geomstats coding style guide](#geomstats-coding-style-guide).**

- Create your feature branch (`git checkout -b feature/fooBar`)
- Commit your changes (`git commit -am 'Add some fooBar'`)
- Push to the branch (`git push origin feature/fooBar`)

Try to keep the changes focused.
If you feel tempted to "branch out" then please make a [new branch][link_branches].

If you are adding a new feature, don't forget to add the corresponding [unit tests][link_unit_tests].
As ``geomstats`` enables numpy and tensorflow, your unit tests should run on these two backends.

**4. Submit a [pull request][link_pullrequest].**

A member of the development team will review your changes to confirm
that they can be merged into the main code base.

Pull requests titles should begin with a descriptive prefix:

* ``ENH``: enhancements or new features
* ``FIX``: bug fixes
* ``TST``: new or updated tests
* ``DOC``: new or updated documentation
* ``STY``: style changes
* ``REF``: refactoring existing code
* ``CI``: updates to continous integration infrastructure
* ``MAINT``: general maintenance

For example: `[ENH] Implementation of Grassmanians`.

For works-in-progress, add the ``WIP`` tag in addition to the descriptive prefix.
Pull-requests tagged with ``[WIP]`` will not be merged until the tag is removed.

## Geomstats Coding Style Guide

Ensure that your code is compliant with [PEP8][link_pep8],
the coding style guide for python, as enforced by [flake8][link_flake8].

Flake8 is a tool, called a [linter][link_linters] that automatically enforces this style.

## Recognizing Contributions

We welcome and recognize all contributions from documentation to testing to code development.
You can see a list of current contributors in our [README.md][link_readme].
If you are new to the project, don't forget to add your name and affiliation there!

## Thank You!

<br>

*&mdash; Based on contributing guidelines from the [fMRIprep][link_fmriprep] project.*

[link_github]: https://github.com/
[link_geomstats]: https://github.com/geomstats/geomstats
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account

[link_git]: https://git-scm.com/
[link_handbook]: https://guides.github.com/introduction/git-handbook/
[link_swc_intro]: http://swcarpentry.github.io/git-novice/

[writing_formatting_github]: https://help.github.com/articles/getting-started-with-writing-and-formatting-on-github
[markdown]: https://daringfireball.net/projects/markdown
[rick_roll]: https://www.youtube.com/watch?v=dQw4w9WgXcQ

[link_issues]: https://github.com/geomstats/geomstats/issues
[link_labels]: https://github.com/geomstats/geomstats/labels
[link_discussingissues]: https://help.github.com/articles/discussing-projects-in-issues-and-pull-requests

[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_unit_tests]: https://github.com/geomstats/geomstats/tree/master/tests
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_pep8]: https://www.python.org/dev/peps/pep-0008/
[link_linters]: https://en.wikipedia.org/wiki/Lint_(software)
[link_flake8]: http://flake8.pycqa.org/en/latest/
[link_readme]: https://github.com/geomstats/geomstats/blob/master/README.md
[link_fmriprep]: https://github.com/poldracklab/fmriprep/
