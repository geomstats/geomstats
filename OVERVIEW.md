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
+ 2D-arrays for _matrix-embedded_ spaces,
+ 1D-arrays for _vector-embedded_ spaces.

The first group for instance contains the space of 
[SPD](geomstats/geometry/spd_matrices_space.py) matrices, 
while the [hyperbolic space](geomstats/geometry/hyperbolic_space.py)
belongs to the second. 

In the following, we summarize the interface that should be shared by geomstat's 
manifold classes. 
These common methods may then be relied upon by any abstract layer 
of optimisation algorithms, such as geomstat's learning module.

### Matrix Spaces and Lie Groups

Inheritance diagram:

```
                               .--- SPD(n)                  
    Mat(n,n) <----- GL(n) <---(                             
                               `--- SO(n)                   
       ^                                                   
       '                                                  
       '                                                    

    Aff(n-1) <----- GA(n) <--------- SE(n)                       
                                                            
```
__Note__ : 
Other Lie groups such as `SL(n)` or `Sp(2n)` may also be implemented in the future.

#### Matrices

implements:
-  elementary matrix operations: 
    + `equal : point -> bool`
    + `mul : (...points) -> point`
    + `transpose : point -> point`
- convenience methods:
    + `is_symmetric : point -> bool`
    + `to_symmetric : point -> point` 

__Note__ : 
an `apply : (linear, vector) -> vector` method
would also be convenient to couple matrix-classes and vector-classes, 
e.g. have SO(n+1) act on the n-Sphere.


#### GeneralLinear 

implements:
- elementary group operations:
    + `identity : () -> point`
    + `compose : (...points) -> point` alias of `mul`
    + `inv : point -> point`
- the Lie group operations: 
    + `exp : (vector, point1) -> point2`
    + `log : (point2, point1) -> vector`
- the interpolating one-parameter orbit:
    + `orbit : (point2, point1) -> (t -> point)`

#### SpecialOrthogonal

overrides: 
+ `inv`: call `transpose`

#### SPD 

overrides:
+ `exp`  
+ `log`: solve the eigenvalue problem. 

__Note__ : 
the symmetry check should be moved from the backend to the SPD group class. 

### Affine 

__Note__ : 
affine transformations are not yet implemented at the moment.  

The algebra of affine transformations on an n-dimensional space
can be represented by square matrices of size n+1, 
i.e. Aff(n) viewed as a subalgebra of Mat(n+1). 

The affine transformation `x -> l(x) + v` is represented by the matrix: 
```
[[l_11, ..., l_1n, v_1],
 [l_21, ..., l_2n, v_2],
  ...
 [l_n1, ..., l_nn, v_n],
 [   0, ...,    0,   1]]
```

inherit:
+ `mul`
+ `equal`
override:
+ `transpose`: restrict to the linear part,
implement: 
+ `to_linear : (affine) -> linear`
+ `to_vector : (affine) -> vector`
+ `apply : (affine, vector) -> vector`

### GeneralAffine

inherit: 
+ `exp`
+ `log`: coincide with the linear matrix operations. 

### SpecialEuclidian

override:
+ `inv`: transpose the linear part and invert the translation vector. 

### Vector Spaces and Embedded Manifolds

...to come...

+ riemannian structures:
    - `exp`
    - `log`
    - `geodesic`
    - `inner`
+ embedding: 
    - `is_tangent : vector -> bool`
    - `to_tangent : vector -> vector`

__Note__ :
`exp` and `log` may more generally derive from a connection. 
In this case the `geodesic` is somewhat confusing 
