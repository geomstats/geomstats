<!--
Thank you for opening this pull request!
-->

## Checklist

- [ ] My pull request has a clear and explanatory title.
- [ ] If necessary, my code is [vectorized](https://www.geeksforgeeks.org/vectorization-in-python/).
- [ ] I added appropriate unit tests.
- [ ] I made sure the code passes all unit tests. (refer to comment below)
- [ ] My PR follows [PEP8](https://peps.python.org/pep-0008/) guidelines. (refer to comment below)
- [ ] My PR follows [geomstats coding style](https://github.com/geomstats/geomstats/blob/main/docs/contributing.rst#coding-style-guidelines) and API.
- [ ] My code is properly documented and I made sure the documentation renders properly. ([Link](https://github.com/geomstats/geomstats/blob/main/docs/contributing.rst#documentation))
- [ ] I linked to issues and PRs that are relevant to this PR.


<!-- For checking consistency of entire codebase
First, run the tests related to your changes. For example, if you changed something in geomstats/spd_matrices_space.py:
$ pytest tests/tests_geomstats/test_spd_matrices.py

and then run the tests of the whole codebase to check that your feature is not breaking any of them:
$ pytest tests/

This way, further modifications on the code base are guaranteed to be consistent with the desired behavior. Merging your PR should not break any test in any backend (numpy, tensorflow or pytorch)."

For testing in alternative backends such as `numpy`, `pytorch`, `autograd`, `tensorflow` set the environment variable using:
$ export GEOMSTATS_BACKEND=<backend_name>

Next, import the `backend` module using:
import geomstats.backend as gs
-->


<!-- For flake8 tests
Install dependencies
$ pip3 install -r dev-requirements.txt

Then run the following commands:
$ flake8 --ignore=D,W503,W504 geomstats examples tests   #shadows .flake8
$ flake8 geomstats/geometry geomstats/learning           #passed two subfolders
-->

## Description

<!-- Include a description of your pull request. If relevant, feel free to use this space to talk about time and space complexity as well scalability of your code-->

## Issue

<!-- Tell us which issue does this PR fix . Why this feature implementation/fix is important in practice ?-->

## Additional context

<!-- Add any extra information -->
