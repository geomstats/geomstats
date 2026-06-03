.. _testing-reference:

======================
Testing Quick Reference
======================

Quick reference guide and cheat sheet for testing in geomstats.

.. contents:: Table of Contents
   :local:


Quick Decision Tree
===================

**What kind of test do I need?**

.. code-block:: text

   Are you fixing a bug or adding a small feature?
   └─→ YES → Write a simple test (see Simple Tests)
   └─→ NO ↓

   Are you adding a new geometric space with standard operations?
   └─→ YES → Use the three-layer architecture (see Advanced Tests)
   └─→ NO ↓

   Are you testing vectorization/batch operations?
   └─→ YES → Use the three-layer architecture
   └─→ NO → Write a simple test


Running Tests
=============

.. code-block:: bash

   # Run specific test
   pytest tests/tests_geomstats/test_geometry/test_my_space.py::test_my_function

   # Run all tests in a file
   pytest tests/tests_geomstats/test_geometry/test_my_space.py

   # Run all geometry tests
   pytest tests/tests_geomstats/test_geometry/

   # Run only smoke tests (fast)
   pytest -m smoke

   # Run with verbose output
   pytest -v tests/tests_geomstats/

   # Run with specific backend
   GEOMSTATS_BACKEND=pytorch pytest tests/

   # Run with coverage
   pytest --cov=geomstats tests/

   # Run and show print statements
   pytest -s tests/tests_geomstats/test_geometry/test_my_space.py


Simple Tests Template
=====================

**When:** Bug fixes, edge cases, simple features

**Where:** ``tests/tests_geomstats/test_geometry/test_my_space.py``

.. code-block:: python

   import pytest
   import geomstats.backend as gs
   from geomstats.geometry.my_space import MySpace

   def test_my_simple_function():
       """Test description."""
       space = MySpace(dim=3)
       point = gs.array([1.0, 2.0, 3.0])

       result = space.my_function(point)
       expected = gs.array([2.0, 4.0, 6.0])

       gs.testing.assert_allclose(result, expected)


   @pytest.mark.parametrize("dim", [2, 3, 5, 10])
   def test_identity_shape(dim):
       """Test with multiple parameters."""
       space = MySpace(dim=dim)
       assert space.identity.shape == (dim, dim)


   @pytest.mark.smoke
   def test_smoke_test():
       """Fast sanity check."""
       space = MySpace()
       assert space.dim > 0


Assertions Cheat Sheet
======================

.. code-block:: python

   # Numerical comparison (preferred for arrays)
   gs.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-5)

   # Exact equality check
   assert gs.all(result == expected)

   # Boolean checks
   assert result is True
   assert result is False

   # Shape checks
   assert result.shape == (3, 3)
   assert result.ndim == 2

   # Type checks
   assert isinstance(result, gs.ndarray)

   # Exceptions
   with pytest.raises(ValueError):
       space.invalid_operation()

   # Approximate scalar
   assert abs(result - expected) < 1e-6


Common Test Markers
===================

.. code-block:: python

   @pytest.mark.smoke          # Fast, basic test
   @pytest.mark.random         # Uses random data
   @pytest.mark.slow           # Expensive test
   @pytest.mark.vec            # Vectorization test
   @pytest.mark.mathprop       # Mathematical property
   @pytest.mark.skip(reason="...") # Skip this test
   @pytest.mark.xfail(reason="...") # Expected failure

   # Backend-specific (from geomstats.test.test_case)
   @np_only                    # NumPy only
   @torch_only                 # PyTorch only
   @autograd_only              # Autograd only


Testing Patterns
================

Test a Mathematical Property
----------------------------

.. code-block:: python

   @pytest.mark.mathprop
   def test_distance_is_symmetric():
       """Test d(a,b) == d(b,a)."""
       space = MySpace()
       point_a = space.random_point()
       point_b = space.random_point()

       dist_ab = space.distance(point_a, point_b)
       dist_ba = space.distance(point_b, point_a)

       gs.testing.assert_allclose(dist_ab, dist_ba)


Test Inverse Operations
-----------------------

.. code-block:: python

   def test_exp_log_inverse():
       """Test exp(log(x)) == x."""
       space = MySpace()
       base_point = space.random_point()
       point = space.random_point()

       log_point = space.log(point, base_point)
       exp_log_point = space.exp(log_point, base_point)

       gs.testing.assert_allclose(exp_log_point, point, atol=1e-6)


Test Edge Cases
---------------

.. code-block:: python

   def test_distance_to_self_is_zero():
       """Distance from point to itself is zero."""
       space = MySpace()
       point = space.random_point()

       distance = space.distance(point, point)

       gs.testing.assert_allclose(distance, 0.0, atol=1e-10)


Test with Random Points
-----------------------

.. code-block:: python

   @pytest.mark.random
   def test_random_point_belongs():
       """Random points should belong to the space."""
       space = MySpace()

       for _ in range(10):
           point = space.random_point()
           assert space.belongs(point)


Three-Layer Architecture Quick Reference
=========================================

File Structure
--------------

.. code-block:: text

   tests/tests_geomstats/test_geometry/
   ├── test_my_space.py           # Layer 3: Concrete tests
   └── data/
       └── my_space.py             # Layer 1: Test data

   geomstats/test_cases/geometry/
   └── my_space.py                 # Layer 2: Test cases


Layer 1: Test Data Template
----------------------------

**File:** ``tests/tests_geomstats/test_geometry/data/my_space.py``

.. code-block:: python

   from geomstats.test.data import TestData

   class MySpaceTestData(TestData):
       """Test data for MySpace."""

       # Configuration
       trials = 3
       tolerances = {
           "my_method": {"atol": 1e-5},
       }
       skips = ("optional_method",)

       def my_method_test_data(self):
           """Test data for my_method."""
           data = [
               dict(input_val=x, expected=y),
               dict(input_val=x2, expected=y2),
           ]
           return self.generate_tests(data)

       def my_method_vec_test_data(self):
           """Vectorization test data."""
           return self.generate_vec_data()

       def my_random_test_test_data(self):
           """Random test data."""
           return self.generate_random_data()


Layer 2: Test Case Template
----------------------------

**File:** ``geomstats/test_cases/geometry/my_space.py``

.. code-block:: python

   from geomstats.test_cases.geometry.base import OpenSetTestCase

   class MySpaceTestCase(OpenSetTestCase):
       """Test case for MySpace."""

       def test_my_method(self, input_val, expected, atol):
           """Test my_method."""
           result = self.space.my_method(input_val)
           self.assertAllClose(result, expected, atol=atol)

       @pytest.mark.vec
       def test_my_method_vec(self, n_reps, atol):
           """Vectorization test."""
           input_val = self._get_random_input()
           expected = self.space.my_method(input_val)

           vec_data = generate_vectorization_data(
               data=[dict(input_val=input_val, expected=expected, atol=atol)],
               arg_names=["input_val"],
               expected_name="expected",
               n_reps=n_reps,
           )
           self._test_vectorization(vec_data)

       @pytest.mark.random
       def test_my_random_test(self, n_points, atol):
           """Random test."""
           points = self.space.random_point(n_points)
           # Test logic...


Layer 3: Concrete Test Template
--------------------------------

**File:** ``tests/tests_geomstats/test_geometry/test_my_space.py``

.. code-block:: python

   import pytest
   from geomstats.geometry.my_space import MySpace
   from geomstats.test.parametrizers import DataBasedParametrizer
   from geomstats.test_cases.geometry.my_space import MySpaceTestCase
   from .data.my_space import MySpaceTestData

   @pytest.mark.smoke
   class TestMySpace(MySpaceTestCase, metaclass=DataBasedParametrizer):
       """Concrete tests for MySpace."""
       space = MySpace(dim=3)
       testing_data = MySpaceTestData()


Common TestData Methods
=======================

.. code-block:: python

   # Basic test generation
   self.generate_tests(data)

   # Random tests
   self.generate_random_data()

   # Vectorization tests
   self.generate_vec_data()
   self.generate_vec_data(vec_type="sym")      # All combinations
   self.generate_vec_data(vec_type="basic")    # Basic vectorization
   self.generate_vec_data(vec_type="repeat-0-2")  # Specific args

   # Shape tests
   self.generate_shape_data()


Test Case Base Classes
=======================

Choose the appropriate base class:

.. code-block:: python

   from geomstats.test_cases.geometry.base import (
       TestCase,              # Base for all tests
       OpenSetTestCase,       # For open sets
       LevelSetTestCase,      # For level sets (manifolds)
   )

   from geomstats.test_cases.geometry.lie_group import (
       LieGroupTestCase,      # For Lie groups
       MatrixLieGroupTestCase,  # For matrix Lie groups
   )

   from geomstats.test_cases.geometry.manifold import (
       ManifoldTestCase,      # For general manifolds
   )

   from geomstats.test_cases.geometry.riemannian_metric import (
       RiemannianMetricTestCase,  # For metrics
   )


Debugging Failed Tests
======================

Read the Error Message
----------------------

.. code-block:: bash

   FAILED test_my_space.py::TestMySpace::test_belongs[point0]

   # [point0] tells you which parameter set failed


Run Specific Parameter Set
---------------------------

.. code-block:: bash

   # Run just the failing parameter
   pytest test_my_space.py::TestMySpace::test_belongs[point0]

   # Show more details
   pytest -vv test_my_space.py::TestMySpace::test_belongs[point0]


Add Print Statements
--------------------

.. code-block:: python

   def test_my_method(self, input_val, expected, atol):
       print(f"Testing with input: {input_val}")
       print(f"Expected: {expected}")
       result = self.space.my_method(input_val)
       print(f"Got: {result}")
       self.assertAllClose(result, expected, atol=atol)

Run with:

.. code-block:: bash

   pytest -s test_my_space.py::TestMySpace::test_my_method


Check Test Discovery
---------------------

.. code-block:: bash

   # See all tests that will run
   pytest --collect-only tests/tests_geomstats/test_geometry/test_my_space.py


Common Errors
=============

Method Name Mismatch
--------------------

.. code-block:: python

   # Error: test_belongs exists but belongs_test_data is missing
   # Fix: Add belongs_test_data() method to your TestData class


Parameter Name Mismatch
-----------------------

.. code-block:: python

   # Error: test_belongs(point, expected) but data has {"pt": ..., "exp": ...}
   # Fix: Match parameter names exactly

   # Test method:
   def test_belongs(self, point, expected, atol):
       pass

   # Data method:
   def belongs_test_data(self):
       return [dict(point=p, expected=e)]  # Names must match!


Missing self.space
------------------

.. code-block:: python

   # Error: AttributeError: 'TestMySpace' object has no attribute 'space'
   # Fix: Define space in your concrete test class

   class TestMySpace(MySpaceTestCase, metaclass=DataBasedParametrizer):
       space = MySpace()  # Don't forget this!
       testing_data = MySpaceTestData()


Import Errors
-------------

.. code-block:: python

   # Always use geomstats.backend
   import geomstats.backend as gs  # Good

   import numpy as np  # Bad - not backend-agnostic


Coverage Requirements
=====================

Geomstats requires **90% coverage** for new code:

.. code-block:: bash

   # Check coverage
   pytest --cov=geomstats --cov-report=term-missing tests/

   # Generate HTML report
   pytest --cov=geomstats --cov-report=html tests/
   # Open htmlcov/index.html in browser


What to Test
------------

✅ Test these:

- Public methods (not starting with ``_``)
- Different parameter combinations
- Edge cases (zero, identity, boundary values)
- Error handling
- Mathematical properties

❌ Don't test these:

- Private methods (``_method_name``)
- Third-party library code
- Simple getters/setters


Complete Example
================

Here's a complete example combining everything:

.. code-block:: python

   # tests/tests_geomstats/test_geometry/data/circle.py
   from geomstats.test.data import TestData
   import geomstats.backend as gs

   class CircleTestData(TestData):
       tolerances = {"belongs": {"atol": 1e-5}}

       def belongs_test_data(self):
           data = [
               dict(point=gs.array([1.0, 0.0]), expected=True),
               dict(point=gs.array([0.0, 1.0]), expected=True),
               dict(point=gs.array([2.0, 0.0]), expected=False),
           ]
           return self.generate_tests(data)

       def belongs_vec_test_data(self):
           return self.generate_vec_data()


   # geomstats/test_cases/geometry/circle.py
   from geomstats.test_cases.geometry.base import LevelSetTestCase

   class CircleTestCase(LevelSetTestCase):
       def test_belongs(self, point, expected, atol):
           result = self.space.belongs(point)
           self.assertAllClose(result, expected, atol=atol)


   # tests/tests_geomstats/test_geometry/test_circle.py
   import pytest
   from geomstats.geometry.circle import Circle
   from geomstats.test.parametrizers import DataBasedParametrizer
   from geomstats.test_cases.geometry.circle import CircleTestCase
   from .data.circle import CircleTestData

   @pytest.mark.smoke
   class TestCircle(CircleTestCase, metaclass=DataBasedParametrizer):
       space = Circle()
       testing_data = CircleTestData()

       # Can still add simple tests directly
       def test_dimension_is_one(self):
           assert self.space.dim == 1


Further Resources
=================

- :ref:`testing-guide` - Detailed beginner's guide
- :ref:`testing-architecture` - In-depth architecture explanation
- :ref:`contributing` - General contribution guidelines
- Existing tests - Browse ``tests/tests_geomstats/`` for examples


Checklist Before Submitting
============================

- [ ] Tests pass locally: ``pytest tests/``
- [ ] Tests pass on all backends (if applicable)
- [ ] Coverage is at least 90%: ``pytest --cov=geomstats``
- [ ] Code follows PEP8: ``ruff check .``
- [ ] Docstrings are complete
- [ ] Tests have descriptive names and docstrings
- [ ] Committed test data if using three-layer architecture
- [ ] Added markers (@pytest.mark.smoke, etc.) where appropriate
