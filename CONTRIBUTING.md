# Contributing to fMRIPrep

Welcome to the fMRIPrep repository!
We're excited you're here and want to contribute.

## Practical guide to submitting your contribution

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below,
please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in.
Here are some [instructions][link_signupinstructions].

Already know what you're looking for in this guide? Jump to the following sections:

* [Joining the conversation](#joining-the-conversation)
* [Contributing through Github](#contributing-through-github)
* [Understanding issues](#understanding-issues)
* [Making a change](#making-a-change)
* [Structuring contributions](#fMRIPrep-coding-style-guide)
* [Licensing](#licensing)
* [Recognizing contributors](#recognizing-contributions)


## Joining the conversation

`fMRIPrep` is maintained by a growing group of enthusiastic developers&mdash;
and we're excited to have you join!
Most of our discussions will take place on open [issues][link_issues].

We also encourage users to report any difficulties they encounter on [NeuroStars][link_stars],
a community platform for discussing neuroimaging.

We actively monitor both spaces and look forward to hearing from you in either venue!


## Contributing through GitHub

[git][link_git] is a really useful tool for version control.
[GitHub][link_github] sits on top of git and supports collaborative and distributed working.

If you're not yet familiar with `git`, there are lots of great resources to help you *git* started!
Some of our favorites include the [git Handbook][link_handbook] and
the [Software Carpentry introduction to git][link_swc_intro].

On GitHub, You'll use [Markdown][markdown] to chat in issues and pull requests.
You can think of Markdown as a few little symbols around your text that will allow GitHub
to render the text with a little bit of formatting.
For example you could write words as bold (`**bold**`), or in italics (`*italics*`),
or as a [link][rick_roll] (`[link](https://https://youtu.be/dQw4w9WgXcQ)`) to another webpage.

GitHub has a really helpful page for getting started with
[writing and formatting Markdown on GitHub][writing_formatting_github].


## Understanding issues

Every project on GitHub uses [issues][link_issues] slightly differently.

The following outlines how the ``fMRIPrep`` developers think about these tools.

* **Issues** are individual pieces of work that need to be completed to move the project forwards.
A general guideline: if you find yourself tempted to write a great big issue that
is difficult to describe as one unit of work, please consider splitting it into two or more issues.

    Issues are assigned [labels](#issue-labels) which explain how they relate to the overall project's
    goals and immediate next steps.


### Issue Labels

The current list of issue labels are [here][link_labels] and include:

* [![Help Wanted](https://img.shields.io/badge/-help%20wanted-159818.svg)][link_helpwanted] *These issues contain a task that a member of the team has determined we need additional help with.*

    If you feel that you can contribute to one of these issues,
    we especially encourage you to do so!

* [![Bugs](https://img.shields.io/badge/-bugs-fc2929.svg)][link_bugs] *These issues point to problems in the project.*

    If you find new a bug, please give as much detail as possible in your issue,
    including steps to recreate the error.
    If you experience the same bug as one already listed,
    please add any additional information that you have as a comment.

* [![Feature](https://img.shields.io/badge/-feature-0052cc.svg)][link_feature] *These issues are asking for new features to be added to the project.*

    Please try to make sure that your requested feature is distinct from any others
    that have already been requested or implemented.
    If you find one that's similar but there are subtle differences,
    please reference the other request in your issue.


## Making a change

We appreciate all contributions to ``fMRIPrep``,
but those accepted fastest will follow a workflow similar to the following:

**1. Comment on an existing issue or open a new issue referencing your addition.**

This allows other members of the ``fMRIPrep`` development team to confirm that you aren't
overlapping with work that's currently underway and that everyone is on the same page
with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front
is so useful to everyone involved.

**2. [Fork][link_fork] the [fMRIPrep repository][link_fMRIPrep] to your profile.**

This is now your own unique copy of ``fMRIPrep``.
Changes here won't effect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date][link_updateupstreamwiki] with the master repository.

**3. Make the changes you've discussed, following the [fMRIPrep coding style guide](#fMRIPrep-coding-style-guide).**

Try to keep the changes focused.
If you feel tempted to "branch out" then please make a [new branch][link_branches].
It can also be helpful to test your changes locally,
using an [fMRIPrep development environment][link_devel].

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

For example: `[ENH] Support for SB-reference in multi-band datasets`

For works-in-progress, add the ``WIP`` tag in addition to the descriptive prefix.
Pull-requests tagged with ``[WIP]`` will not be merged until the tag is removed.

## fMRIPrep coding style guide

Whenever possible, instances of Nodes and Workflows should use the same names
as the variables they are assigned to.
This makes it easier to relate the content of the working directory to the code
that generated it when debugging.

Workflow variables should end in `_wf` to indicate that they refer to Workflows
and not Nodes.
For instance, a workflow whose basename is `myworkflow` might be defined as
follows:

```Python
from nipype.pipeline import engine as pe

myworkflow_wf = pe.Workflow(name='myworkflow_wf')
```

If a workflow is generated by a function, the name of the function should take
the form `init_<basename>_wf`:

```Python
def init_myworkflow_wf(name='myworkflow_wf):
    workflow = pe.Workflow(name=name)
    ...
    return workflow

myworkflow_wf = init_workflow_wf(name='myworkflow_wf')
```

If multiple instances of the same workflow might be instantiated in the same
namespace, the workflow names and variables should include either a numeric
identifier or a one-word description, such as:

```Python
myworkflow0_wf = init_workflow_wf(name='myworkflow0_wf')
myworkflow1_wf = init_workflow_wf(name='myworkflow1_wf')

# or

myworkflow_lh_wf = init_workflow_wf(name='myworkflow_lh_wf')
myworkflow_rh_wf = init_workflow_wf(name='myworkflow_rh_wf')
```


## Licensing

``fMRIPrep`` is licensed under the BSD 3-clause license.
By contributing to `fMRIPrep`,
you acknowledge that any contributions will be licensed under the same terms.


## Recognizing contributions

We welcome and recognize all contributions from documentation to testing to code development.
You can see a list of current contributors in our [zenodo file][link_zenodo].
If you are new to the project, don't forget to add your name and affiliation there!

## Thank you!

You're awesome. :wave::smiley:

<br>

*&mdash; Based on contributing guidelines from the [STEMMRoleModels][link_stemmrolemodels] project.*

[link_github]: https://github.com/
[link_fMRIPrep]: https://github.com/poldracklab/fmriprep
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account

[link_stars]: https://neurostars.org/

[link_git]: https://git-scm.com/
[link_handbook]: https://guides.github.com/introduction/git-handbook/
[link_swc_intro]: http://swcarpentry.github.io/git-novice/

[writing_formatting_github]: https://help.github.com/articles/getting-started-with-writing-and-formatting-on-github
[markdown]: https://daringfireball.net/projects/markdown
[rick_roll]: https://www.youtube.com/watch?v=dQw4w9WgXcQ

[link_issues]: https://github.com/poldracklab/fmriprep/issues
[link_labels]: https://github.com/poldracklab/fmriprep/labels
[link_discussingissues]: https://help.github.com/articles/discussing-projects-in-issues-and-pull-requests

[link_bugs]: https://github.com/poldracklab/fmriprep/labels/bug
[link_helpwanted]: https://github.com/poldracklab/fmriprep/labels/help%20wanted
[link_feature]: https://github.com/poldracklab/fmriprep/labels/feature

[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_stemmrolemodels]: https://github.com/KirstieJane/STEMMRoleModels
[link_zenodo]: https://github.com/poldracklab/fmriprep/blob/master/.zenodo.json
[link_devel]: https://fmriprep.readthedocs.io/en/latest/contributors.html

