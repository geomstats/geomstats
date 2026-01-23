.. _testing-guide:

========================
Testing Guide for Geomstats
========================

This guide will help you write tests for geomstats, whether you're fixing a bug,
adding a new feature, or contributing for the first time.

.. contents:: Table of Contents
   :local:
   :depth: 2


Quick Start: Your First Test
=============================

If you need to write a simple test (e.g., for a bug fix or small feature),
you can write it directly without understanding the full testing infrastructure.

Basic Example
-------------

Create or find the test file corresponding to your module:

.. code-block:: python

   # In tests/tests_geomstats/test_geometry/test_my_space.py

   import pytest
   import geomstats.backend as gs
   from geomstats.geometry.my_space import MySpace

   def test_my_bugfix():
       """Test that issue #123 is fixed."""
       space = MySpace(dim=3)
       point = gs.array([1.0, 2.0, 3.0])

       result = space.my_function(point)
       expected = gs.array([2.0, 4.0, 6.0])

       gs.testing.assert_allclose(result, expected)

Run your test:

.. code-block:: bash

   $ pytest tests/tests_geomstats/test_geometry/test_my_space.py::test_my_bugfix


When to Use Simple Tests
-------------------------

Use simple, direct tests for:

- **Bug fixes** - Add a regression test that would have caught the bug
- **Edge cases** - Test specific corner cases you discovered
- **Quick validation** - Test a specific behavior or property
- **Learning** - When you're new to the project

These tests don't need vectorization, parametrization, or the data-driven
architecture. Just write clear, focused tests.


Understanding Test Organization
================================

Geomstats has 110+ test files organized by domain:

.. code-block:: text

   tests/tests_geomstats/
   ├── test_geometry/          # Tests for manifolds and spaces
   │   ├── test_euclidean.py
   │   ├── test_hypersphere.py
   │   ├── test_special_orthogonal.py
   │   └── data/               # Test data for parametrized tests
   │       ├── euclidean.py
   │       └── special_orthogonal.py
   ├── test_learning/          # Machine learning algorithms
   ├── test_distributions/     # Probability distributions
   └── test_numerics/          # Numerical methods

Finding the Right Test File
----------------------------

**Rule of thumb:** If you modified ``geomstats/geometry/my_space.py``,
add tests to ``tests/tests_geomstats/test_geometry/test_my_space.py``.

If the test file doesn't exist, create it following the naming convention:
``test_<module_name>.py``


Writing Tests: Step by Step
=============================

Step 1: Import What You Need
-----------------------------

.. code-block:: python

   import pytest
   import geomstats.backend as gs
   from geomstats.geometry.my_space import MySpace

- Always use ``geomstats.backend as gs`` for backend compatibility
- Import the classes/functions you're testing


Step 2: Write a Test Function
------------------------------

Test function names must start with ``test_``:

.. code-block:: python

   def test_belongs_returns_true_for_valid_point():
       """Test that belongs() returns True for points on the manifold."""
       space = MySpace(dim=3)
       point = space.random_point()

       result = space.belongs(point)

       assert result is True

**Good practices:**

- Use descriptive names: ``test_belongs_returns_true_for_valid_point``
  rather than ``test_1``
- Add a docstring explaining what you're testing
- One assertion per test (when possible)
- Test one behavior at a time


Step 3: Use Appropriate Assertions
-----------------------------------

For backend-compatible assertions, use:

.. code-block:: python

   # For numerical comparisons (handles floating point)
   gs.testing.assert_allclose(result, expected, atol=1e-6)

   # For exact equality
   assert gs.all(result == expected)

   # For boolean results
   assert result is True
   assert result is False

   # For shapes
   assert result.shape == (3, 3)

**Don't use** ``==`` directly for arrays, as it won't work correctly
with PyTorch tensors.


Step 4: Add Test Markers
-------------------------

Use pytest markers to categorize your tests:

.. code-block:: python

   @pytest.mark.smoke
   def test_identity_is_identity():
       """Smoke test: quick sanity check."""
       space = MySpace()
       identity = space.identity
       assert gs.all(identity == gs.eye(3))

   @pytest.mark.random
   def test_exp_log_inverse(n_points=5):
       """Test with random data."""
       space = MySpace()
       points = space.random_point(n_points)
       # ... test logic

   @pytest.mark.slow
   def test_expensive_computation():
       """Mark slow tests to skip in quick test runs."""
       # ... expensive test

**Common markers:**

- ``@pytest.mark.smoke`` - Fast, basic tests for CI
- ``@pytest.mark.random`` - Tests using random data
- ``@pytest.mark.slow`` - Slow tests (run less frequently)
- ``@pytest.mark.vec`` - Vectorization tests (auto-generated)


Running Your Tests
==================

Run a specific test:

.. code-block:: bash

   $ pytest tests/tests_geomstats/test_geometry/test_my_space.py::test_my_function

Run all tests in a file:

.. code-block:: bash

   $ pytest tests/tests_geomstats/test_geometry/test_my_space.py

Run all tests in a directory:

.. code-block:: bash

   $ pytest tests/tests_geomstats/test_geometry/

Run only smoke tests (fast):

.. code-block:: bash

   $ pytest -m smoke

Run with verbose output:

.. code-block:: bash

   $ pytest -v tests/tests_geomstats/test_geometry/test_my_space.py

Run with backend selection:

.. code-block:: bash

   $ GEOMSTATS_BACKEND=pytorch pytest tests/tests_geomstats/test_geometry/test_my_space.py


Testing Best Practices
=======================

Test What Could Go Wrong
-------------------------

Think about edge cases:

.. code-block:: python

   def test_distance_is_zero_for_same_point():
       """Distance from a point to itself should be zero."""
       space = MySpace()
       point = space.random_point()

       distance = space.distance(point, point)

       gs.testing.assert_allclose(distance, 0.0, atol=1e-10)

   def test_distance_is_symmetric():
       """Distance should be symmetric: d(a,b) = d(b,a)."""
       space = MySpace()
       point_a = space.random_point()
       point_b = space.random_point()

       dist_ab = space.distance(point_a, point_b)
       dist_ba = space.distance(point_b, point_a)

       gs.testing.assert_allclose(dist_ab, dist_ba)


Test Mathematical Properties
-----------------------------

For geometric objects, test mathematical properties:

.. code-block:: python

   @pytest.mark.mathprop
   def test_exp_log_composition_is_identity():
       """Test that exp(log(point, base), base) == point."""
       space = MySpace()
       base_point = space.random_point()
       point = space.random_point()

       log_result = space.log(point, base_point)
       exp_result = space.exp(log_result, base_point)

       gs.testing.assert_allclose(exp_result, point, atol=1e-6)


Use Parametrization for Multiple Cases
---------------------------------------

Test multiple scenarios efficiently:

.. code-block:: python

   @pytest.mark.parametrize("dim", [2, 3, 5, 10])
   def test_identity_has_correct_shape(dim):
       """Test identity matrix has shape (dim, dim)."""
       space = MySpace(dim=dim)
       identity = space.identity

       assert identity.shape == (dim, dim)

   @pytest.mark.parametrize("point,expected", [
       (gs.array([1.0, 0.0]), True),
       (gs.array([0.0, 1.0]), True),
       (gs.array([2.0, 0.0]), False),
       (gs.array([0.0, 0.0]), False),
   ])
   def test_belongs_specific_cases(point, expected):
       """Test belongs for specific known cases."""
       space = UnitCircle()
       result = space.belongs(point)
       assert result == expected


Test Coverage Requirements
---------------------------

Geomstats requires **90% test coverage** for new code. This means:

- Every public function should have at least one test
- Test the main code path and error cases
- Don't test private functions (starting with ``_``) directly

Check coverage locally:

.. code-block:: bash

   $ pytest --cov=geomstats tests/tests_geomstats/test_geometry/test_my_space.py
   $ pytest --cov=geomstats --cov-report=html tests/


Common Testing Patterns
=======================

Testing a New Manifold
----------------------

When adding a new manifold, test these core methods:

.. code-block:: python

   def test_belongs():
       """Test that belongs correctly identifies points on the manifold."""
       pass

   def test_random_point_belongs():
       """Test that random_point generates valid points."""
       space = MySpace()
       point = space.random_point()
       assert space.belongs(point)

   def test_projection_belongs():
       """Test that projection returns a point on the manifold."""
       space = MySpace()
       point = gs.random.normal(size=space.shape)
       projected = space.projection(point)
       assert space.belongs(projected)

   def test_dimension():
       """Test the dimension property."""
       space = MySpace(dim=5)
       assert space.dim == 5


Testing a New Metric
---------------------

For Riemannian metrics:

.. code-block:: python

   def test_metric_is_positive():
       """Test that metric(v, v) >= 0."""
       pass

   def test_metric_is_symmetric():
       """Test that metric(u, v) == metric(v, u)."""
       pass

   def test_exp_belongs():
       """Test that exp returns a point on the manifold."""
       pass

   def test_log_belongs():
       """Test that log returns a tangent vector."""
       pass


When Simple Tests Aren't Enough
================================

If you need to test vectorization (batch operations) or need comprehensive
testing across many parameter combinations, see:

- :ref:`testing-architecture` - Understanding the data-driven test infrastructure
- :ref:`testing-reference` - Quick reference for advanced testing patterns

These guides explain the three-layer testing architecture used in geomstats:

1. **Test Data Layer** (in ``data/`` subdirectories) - Defines test cases
2. **Test Case Layer** (in ``geomstats/test_cases/``) - Reusable test logic
3. **Concrete Test Layer** (in ``tests/tests_geomstats/``) - Instantiates tests

This architecture enables automatic vectorization testing and massive code reuse.


Getting Help
============

If you're stuck:

1. Look at similar existing tests for patterns
2. Ask on GitHub issues or discussions
3. Reach out on the mailing list: hi@geomstats.ai
4. Check the :ref:`contributing` guide for general contribution help

Remember: **It's okay to start with simple tests!** You can always refactor
to more sophisticated patterns later. The most important thing is having tests
that validate your code works correctly.
