.. _testing-architecture:

=============================
Testing Architecture
=============================

This document explains geomstats' sophisticated testing infrastructure,
designed for comprehensive testing of geometric operations across multiple
backends with automatic vectorization validation.

.. contents:: Table of Contents
   :local:
   :depth: 3


Overview
========

Geomstats uses a **three-layer testing architecture** that separates:

1. **What to test** (test data)
2. **How to test it** (test case logic)
3. **Where to test it** (concrete test instances)

This architecture enables:

- Testing one implementation across many geometric spaces
- Automatic vectorization testing for batch operations
- Backend-agnostic test definitions (NumPy, PyTorch, Autograd)
- Parametrized testing without code duplication
- Shared test logic across 41+ geometry types


The Three Layers
================

Layer 1: Test Data (What to Test)
----------------------------------

**Location:** ``tests/tests_geomstats/*/data/*.py``

Test data classes define:

- Input values and expected outputs
- Number of test trials
- Tolerances for numerical comparisons
- Which tests to skip or mark as expected failures
- Random data generation parameters

**Example:** ``tests/tests_geomstats/test_geometry/data/special_orthogonal.py``

.. code-block:: python

   from geomstats.test.data import TestData

   class SpecialOrthogonalMatrices2TestData(TestData):
       """Test data for SO(2) matrices."""

       def belongs_test_data(self):
           """Provide test cases for the belongs() method."""
           theta = gs.pi / 3
           data = [
               dict(point=rotation_matrix(theta), expected=True),
               dict(point=gs.zeros((2, 2)), expected=False),
               dict(
                   point=gs.stack([rotation_matrix(theta), gs.zeros((2, 2))]),
                   expected=gs.array([True, False])
               ),
           ]
           return self.generate_tests(data)

       def identity_test_data(self):
           """Test cases for identity element."""
           data = [dict(expected=gs.eye(2))]
           return self.generate_tests(data)

**Key features:**

- Method names match pattern: ``<method_name>_test_data()``
- Returns list of dicts with test parameters
- ``generate_tests()`` applies pytest markers
- Can set class-level ``tolerances``, ``skips``, ``trials``


Layer 2: Test Cases (How to Test)
----------------------------------

**Location:** ``geomstats/test_cases/geometry/*.py``

Test case classes define the actual test logic, reusable across different spaces:

**Example:** ``geomstats/test_cases/geometry/special_orthogonal.py``

.. code-block:: python

   from geomstats.test_cases.geometry.base import LevelSetTestCase

   class SpecialOrthogonalMatricesTestCase(LevelSetTestCase):
       """Reusable tests for special orthogonal matrices."""

       def test_belongs(self, point, expected, atol):
           """Test the belongs method."""
           result = self.space.belongs(point)
           self.assertAllClose(result, expected, atol=atol)

       @pytest.mark.random
       def test_rotation_vector_from_matrix_after_matrix_from_rotation_vector(
           self, n_points, atol
       ):
           """Test that conversions are inverse operations."""
           vec = self._get_random_rotation_vector(n_points)
           mat = self.space.matrix_from_rotation_vector(vec)
           vec_reconstructed = self.space.rotation_vector_from_matrix(mat)
           self.assertAllClose(vec, vec_reconstructed, atol=atol)

**Key features:**

- Test method parameters match data method return values
- Uses ``self.space`` to access the space being tested
- Inherits from base test cases (``TestCase``, ``LieGroupTestCase``, etc.)
- Can use helper methods like ``assertAllClose``


Layer 3: Concrete Tests (Where to Test)
----------------------------------------

**Location:** ``tests/tests_geomstats/test_geometry/*.py``

Concrete test classes instantiate the tests for specific spaces:

**Example:** ``tests/tests_geomstats/test_geometry/test_special_orthogonal.py``

.. code-block:: python

   from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices
   from geomstats.test.parametrizers import DataBasedParametrizer
   from geomstats.test_cases.geometry.special_orthogonal import (
       SpecialOrthogonalMatricesTestCase,
   )
   from .data.special_orthogonal import SpecialOrthogonalMatrices2TestData

   @pytest.mark.smoke
   class TestSpecialOrthogonalMatrices2(
       SpecialOrthogonalMatricesTestCase,  # Layer 2: test logic
       metaclass=DataBasedParametrizer     # Magic: automatic parametrization
   ):
       space = _SpecialOrthogonalMatrices(n=2, equip=False)  # What to test
       testing_data = SpecialOrthogonalMatrices2TestData()   # Layer 1: test data

The ``DataBasedParametrizer`` metaclass automatically:

1. Discovers test methods (e.g., ``test_belongs``)
2. Finds matching data methods (e.g., ``belongs_test_data``)
3. Applies ``@pytest.mark.parametrize`` with the test data
4. Generates vectorization tests automatically


How It Works: The Metaclass Magic
==================================

The Parametrizer Metaclass
---------------------------

**File:** ``geomstats/test/parametrizers.py``

The ``DataBasedParametrizer`` metaclass automatically connects test methods
to their data:

.. code-block:: python

   class DataBasedParametrizer(type):
       """Metaclass that auto-generates parametrized tests.

       For each test method test_foo(), looks for a foo_test_data() method
       in the testing_data class and automatically parametrizes the test.
       """

**How it works:**

1. When a test class is defined, the metaclass inspects it
2. For each ``test_*`` method, it looks for ``*_test_data()`` in ``testing_data``
3. If found, applies ``@pytest.mark.parametrize`` automatically
4. Generates vectorization tests based on configuration

**Example transformation:**

.. code-block:: python

   # What you write:
   class TestMySpace(MyTestCase, metaclass=DataBasedParametrizer):
       space = MySpace()
       testing_data = MyTestData()

   # What the metaclass generates (conceptually):
   class TestMySpace(MyTestCase):
       space = MySpace()

       @pytest.mark.parametrize("point,expected,atol", [
           (data1["point"], data1["expected"], data1.get("atol", 1e-6)),
           (data2["point"], data2["expected"], data2.get("atol", 1e-6)),
       ])
       def test_belongs(self, point, expected, atol):
           result = self.space.belongs(point)
           self.assertAllClose(result, expected, atol=atol)


Test Data Generation
--------------------

**File:** ``geomstats/test/data.py``

The ``TestData`` base class provides helpers for generating test data:

.. code-block:: python

   class TestData:
       """Base class for test data."""

       trials = 3  # Number of random test repetitions
       N_RANDOM_POINTS = [1, 2, 3]  # Points for random tests
       N_VEC_REPS = [1, 2, 3]  # Repetitions for vectorization tests

       def generate_tests(self, data, marks=()):
           """Add pytest markers to test data."""
           # Adds markers, tolerances, returns formatted data

       def generate_random_data(self, marks=()):
           """Generate random test data."""
           data = []
           for n_points in self.N_RANDOM_POINTS:
               for _ in range(self.trials):
                   data.append(dict(n_points=n_points))
           return self.generate_tests(data, marks=marks)

       def generate_vec_data(self, vec_type="sym"):
           """Generate vectorization test data."""
           data = []
           for n_reps in self.N_VEC_REPS:
               data.append(dict(n_reps=n_reps))
           return self.generate_tests(data, marks=(pytest.mark.vec,))


Vectorization Testing
=====================

Why Vectorization Matters
--------------------------

Geomstats is a numerical library where operations must work on:

- Single points: ``shape = (dim,)``
- Batches of points: ``shape = (batch_size, dim)``
- Multiple batches: ``shape = (batch1, batch2, dim)``

**Example:**

.. code-block:: python

   # Single distance
   d = metric.dist(point_a, point_b)  # scalar

   # Batch distances
   d = metric.dist(batch_of_points_a, batch_of_points_b)  # array of distances

Vectorization testing ensures functions handle all these cases correctly.


Automatic Vectorization Tests
------------------------------

**File:** ``geomstats/test/vectorization.py``

The framework automatically generates vectorization tests:

.. code-block:: python

   @pytest.mark.vec
   def test_belongs_vec(self, n_reps, atol):
       """Auto-generated vectorization test for belongs()."""
       # Generate single point
       point = self.space.random_point()
       expected = self.space.belongs(point)

       # Generate vectorized data
       vec_data = generate_vectorization_data(
           data=[dict(point=point, expected=expected, atol=atol)],
           arg_names=["point"],
           expected_name="expected",
           n_reps=n_reps,
       )
       self._test_vectorization(vec_data)

**What it tests:**

- ``n_reps=1``: ``belongs(point)`` vs ``belongs([point])``
- ``n_reps=2``: ``belongs([point1])`` vs ``belongs([point1, point1])``
- ``n_reps=3``: ``belongs([point1, point2])`` vs ``belongs([point1, point1, point2, point2])``

This ensures proper broadcasting behavior.


Backend Compatibility
=====================

Testing Across Backends
------------------------

Geomstats supports three backends:

- NumPy (default)
- PyTorch
- Autograd

Tests must work across all backends. The test infrastructure handles this through:

1. **Backend-agnostic assertions**

.. code-block:: python

   # Good: Works with all backends
   self.assertAllClose(result, expected, atol=1e-6)

   # Bad: NumPy-specific
   np.testing.assert_allclose(result, expected)

2. **Backend markers for conditional tests**

.. code-block:: python

   from geomstats.test.test_case import np_only, torch_only

   @np_only
   def test_numpy_specific_feature(self):
       """Only runs with NumPy backend."""
       pass

   @torch_only
   def test_pytorch_specific_feature(self):
       """Only runs with PyTorch backend."""
       pass

3. **Backend detection**

.. code-block:: python

   if self.np_backend():
       # NumPy-specific logic
       pass
   elif self.pytorch_backend():
       # PyTorch-specific logic
       pass


Test Markers
============

Geomstats uses pytest markers to categorize tests:

.. code-block:: python

   # In pyproject.toml
   markers = [
       "smoke: simple and basic numerical tests.",
       "random: tests that use randomized data.",
       "validation: not smoke, neither random.",
       "vec: vectorization tests.",
       "shape: array shape tests.",
       "type: checks output types.",
       "mathprop: mathematical properties tests.",
       "slow: for slow tests.",
       "redundant: redundant test.",
       "ignore: deselect tests.",
   ]

**Usage:**

.. code-block:: bash

   # Run only smoke tests (fast)
   $ pytest -m smoke

   # Run all except slow tests
   $ pytest -m "not slow"

   # Run smoke and random tests
   $ pytest -m "smoke or random"


Advanced Features
=================

Tolerance Configuration
-----------------------

Different tests may need different numerical tolerances:

.. code-block:: python

   class MyTestData(TestData):
       tolerances = {
           "projection_belongs": {"atol": 1e-5},
           "exp_log_inverse": {"atol": 1e-6, "rtol": 1e-5},
       }

These tolerances are automatically passed to test methods.


Skip Configuration
------------------

Skip tests that aren't applicable:

.. code-block:: python

   class MyTestData(TestData):
       skips = (
           "projection_belongs",  # No projection method
           "distance_is_symmetric",  # Not a metric space
       )


Expected Failures
-----------------

Mark known issues as expected failures:

.. code-block:: python

   class MyTestData(TestData):
       xfails = (
           "exp_belongs",  # Known issue #123
       )


Random Data Generators
----------------------

**File:** ``geomstats/test/random.py``

Specialized random data generators for different spaces:

.. code-block:: python

   from geomstats.test.random import RandomDataGenerator

   class MyRandomDataGenerator(RandomDataGenerator):
       def point(self, n_points=1):
           """Generate random points on the space."""
           return self.space.random_point(n_points)

       def tangent_vec(self, point=None, n_points=1):
           """Generate random tangent vectors."""
           if point is None:
               point = self.point(n_points)
           return self.space.to_tangent(
               gs.random.normal(size=(n_points,) + self.space.shape),
               point
           )


Inheritance Hierarchies
=======================

Test Case Inheritance
---------------------

Test cases follow the inheritance hierarchy of the spaces they test:

.. code-block:: text

   TestCase (base)
   ├── OpenSetTestCase
   │   └── VectorSpaceTestCase
   │       └── EuclideanTestCase
   ├── LevelSetTestCase
   │   └── LieGroupTestCase
   │       ├── MatrixLieGroupTestCase
   │       │   └── SpecialOrthogonalMatricesTestCase
   │       └── SpecialOrthogonalVectorsTestCase
   └── QuotientMetricTestCase

This enables automatic testing of inherited properties.


Test Data Inheritance
---------------------

Test data classes also inherit:

.. code-block:: python

   class LieGroupTestData(TestData):
       """Common tests for all Lie groups."""

       def identity_test_data(self):
           pass

       def compose_test_data(self):
           pass

   class SpecialOrthogonalMatricesTestData(LieGroupTestData):
       """Adds SO(n)-specific tests."""

       def are_antipodals_test_data(self):
           pass


Working with the Architecture
==============================

When to Use the Architecture
----------------------------

Use the three-layer architecture when:

- Adding a new geometric space with standard operations
- Testing operations that need vectorization validation
- Writing tests that apply to multiple similar spaces
- Contributing core geometry functionality

**Don't use it when:**

- Writing a simple bug fix test
- Testing a one-off edge case
- Learning the codebase for the first time


Adding Tests for a New Space
-----------------------------

**Step 1:** Create test data class

.. code-block:: python

   # tests/tests_geomstats/test_geometry/data/my_space.py

   from geomstats.test.data import TestData

   class MySpaceTestData(TestData):
       def belongs_test_data(self):
           data = [
               dict(point=valid_point, expected=True),
               dict(point=invalid_point, expected=False),
           ]
           return self.generate_tests(data)

**Step 2:** Create or reuse test case class

.. code-block:: python

   # geomstats/test_cases/geometry/my_space.py

   from geomstats.test_cases.geometry.base import OpenSetTestCase

   class MySpaceTestCase(OpenSetTestCase):
       def test_my_specific_method(self, input_val, expected, atol):
           result = self.space.my_specific_method(input_val)
           self.assertAllClose(result, expected, atol=atol)

**Step 3:** Create concrete test

.. code-block:: python

   # tests/tests_geomstats/test_geometry/test_my_space.py

   from geomstats.geometry.my_space import MySpace
   from geomstats.test.parametrizers import DataBasedParametrizer
   from geomstats.test_cases.geometry.my_space import MySpaceTestCase
   from .data.my_space import MySpaceTestData

   @pytest.mark.smoke
   class TestMySpace(MySpaceTestCase, metaclass=DataBasedParametrizer):
       space = MySpace(dim=3)
       testing_data = MySpaceTestData()


Debugging Parametrized Tests
-----------------------------

When a parametrized test fails, pytest shows which parameter set failed:

.. code-block:: bash

   FAILED tests/test_geometry/test_my_space.py::TestMySpace::test_belongs[point0] - AssertionError
   FAILED tests/test_geometry/test_my_space.py::TestMySpace::test_belongs[point1] - AssertionError

To run a specific parameter set:

.. code-block:: bash

   $ pytest tests/test_geometry/test_my_space.py::TestMySpace::test_belongs[point0]

To see the actual parameter values, run with ``-vv``:

.. code-block:: bash

   $ pytest -vv tests/test_geometry/test_my_space.py::TestMySpace::test_belongs


Common Pitfalls
===============

Pitfall 1: Method Name Mismatch
--------------------------------

The test method and data method must match:

.. code-block:: python

   # Wrong: Names don't match
   def test_belongs(...):  # test method
       pass

   def belong_test_data(self):  # data method (typo!)
       pass

   # Correct:
   def test_belongs(...):  # test method
       pass

   def belongs_test_data(self):  # data method
       pass


Pitfall 2: Parameter Name Mismatch
-----------------------------------

Test method parameters must match data dict keys:

.. code-block:: python

   # Wrong:
   def test_belongs(self, point, expected_result, atol):  # expected_result
       pass

   def belongs_test_data(self):
       return [dict(point=p, expected=e)]  # expected (mismatch!)

   # Correct:
   def test_belongs(self, point, expected, atol):
       pass

   def belongs_test_data(self):
       return [dict(point=p, expected=e)]


Pitfall 3: Forgetting generate_tests()
---------------------------------------

Always wrap data with ``generate_tests()``:

.. code-block:: python

   # Wrong:
   def belongs_test_data(self):
       return [dict(point=p, expected=e)]  # No markers applied!

   # Correct:
   def belongs_test_data(self):
       data = [dict(point=p, expected=e)]
       return self.generate_tests(data)


Pitfall 4: Not Testing Vectorization
-------------------------------------

Add vectorization test data:

.. code-block:: python

   class MyTestData(TestData):
       def my_method_vec_test_data(self):
           """Vectorization test for my_method."""
           return self.generate_vec_data()


Further Reading
===============

- :ref:`testing-guide` - Beginner's guide to testing
- :ref:`testing-reference` - Quick reference and cheat sheet
- ``geomstats/test/README.md`` - Additional developer notes
- :ref:`contributing` - General contribution guidelines


Summary
=======

The three-layer architecture may seem complex initially, but it provides:

1. **Massive code reuse** - One test case, many instantiations
2. **Automatic vectorization testing** - Ensures batch operations work
3. **Clear separation of concerns** - What vs how vs where
4. **Maintainability** - Changes propagate automatically
5. **Consistency** - Enforces testing patterns across the codebase

For simple contributions, you don't need to use this architecture.
But when adding comprehensive tests for geometric spaces, it provides
powerful automation and consistency.
