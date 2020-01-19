# Package overview

The geomstats package essentially consists of three layers: 
+ the [backend](geomstats/backend) module 
providing an agnostic interface to numpy, pytorch and tensorflow,
+ the [geometry](geomstats/geometry) module 
implementing a variety of geometric structures, 
+ the [learning](geomstats/learning) module
adapting usual optimisation algorithms to manifolds.

# Classes and inheritance 

The `geometry` module defines a collection of classes 
representing some of the most usual manifolds. 
Manifold objects are usually simply constructed by supplying 
dimension arguments. For instance: 
```
S3 = Hypersphere(3)
```
instantiates a 3-sphere object. 

Operations on manifolds are then exposed as methods of these objects, 
allowing to compute distances, geodesic paths, etc. 
Those that do not depend explicitly on the dimension 
should be implemented as class methods, 
so that they may also be accessed directly 
from the class. 
```
Matrices.mul(a, b, c)
```
returns the product of matrices `a`, `b`, and `c`.
Most methods are vectorized 
and accept arrays of elements, following (and sometimes correcting) 
numpy's broadcasting behaviour.  

Depending on the shape of their elements, 
geomstats manifolds may joined in two groups:
+ _matrix_ spaces, whose methods act on 2D-arrays,
+ _vector_ spaces, whose methods act on 1D-array.  


## Matrices and classical Lie groups





## Vectors and embedded riemannian manifolds

 
