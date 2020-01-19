# Package Overview

The geomstats package essentially consists of three layers: 
+ the [backend](geomstats/backend) module 
providing an agnostic interface to numpy, pytorch and tensorflow,
+ the [geometry](geomstats/geometry) module 
implementing a variety of geometric structures, 
+ the [learning](geomstats/learning) module
adapting usual optimisation algorithms to manifolds.

## Classes and Inheritance 

The `geometry` module defines a collection of classes 
representing some of the most usual manifolds.\\
Manifold objects are usually simply constructed by supplying 
dimension arguments. For instance: 
```python
S3 = Hypersphere(3)
```
instantiates a 3-sphere object. 

Operations on manifolds are exposed as methods of these objects, 
allowing to compute distances, geodesic paths, etc.\\
Those that do not depend explicitly on the dimension 
should be implemented as class methods, 
so that they may also be accessed directly 
from the class. 
```python
Matrices.mul(a, b, c)
```
returns the product of matrices `a`, `b`, and `c`.

Most methods are vectorized 
and accept arrays of elements, following --sometimes fixing--
numpy's broadcasting behaviour.

Depending on the array shape their elementary methods expect, 
geomstats manifolds may be joined in two groups:
+ 2D-arrays for _matrix-embedded spaces_,
+ 1D-arrays for _vector-embedded spaces_.

The first group for instance contains the space of 
[SPD](geomstats/geometry/spd_matrices_space.py) matrices, 
while the [hyperbolic space](geomstats/geometry/hyperbolic_space.py)
belongs to the second. 

In the following, we summarize the interface that should be shared by geomstat's 
manifold classes. 
Some of these common methods may then be relied upon by any abstract layer 
of optimisation algorithms, such as geomstat's learning module.

### Matrix Spaces and Lie Groups



### Vector Spaces and Embedded Manifolds

 
