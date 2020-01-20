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
representing some of the most usual manifolds.
Manifold objects are usually simply constructed by supplying 
dimension arguments. For instance: 
```python
S3 = Hypersphere(3)
```
instantiates a 3-sphere object. 

Operations on manifolds are exposed as methods of these objects, 
allowing to compute distances, geodesic paths, etc.
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

## Matrix Spaces and Lie Groups

Inheritance diagram:

```python
                               .--- SPD(n)                  
    Mat(n,n) <----- GL(n) <---(                             
                               `--- SO(n)                   
       ^                                                  
       |                                                  
                                                           
    Aff(n-1) <----- GA(n) <--------- SE(n)                       
                                                            
```
__Note:__\
Other Lie groups such as `SL(n)` or `Sp(2n)` may also be implemented in the future.

### Matrices

implements:
```python
# elementary matrix operations: 
    equal   : point -> bool
    mul     : (...points) -> point
# to be complemented with?
    sum     : (...points) -> point 
    span    : (scalars, points) -> point
# scalar product and duality:
    transpose       : point -> point
    is_symmetric    : point -> bool
    to_symmetric    : point -> point 
```

__Note:__\
An `apply : (linear, vector) -> vector` method
would also be convenient to couple matrix-classes and vector-classes, 
e.g. have SO(n+1) act on the n-Sphere.


### GeneralLinear 

implements:
```python
# elementary group operations:
    identity: () -> point
    compose : (...points) -> point alias of mul
    inv     : point -> point
# Lie group operations: 
    exp     : (vector, point1) -> point2
    log     : (point2, point1) -> vector
# interpolating one-parameter orbit:
    orbit    : (point2, point1) -> (t -> point)
```

### SpecialOrthogonal

overrides: `inv`, call `transpose`.

### SPDMatrices

overrides: `exp` and `log`, compute eigenvectors, and `compose`.

__Note:__\
The actual symmetry check and Yann `symexp`'s function would 
be moved from the backend to the SPD group class. 

### Affine 

__Note:__ Affine transformations are not implemented at the moment.  

`Aff(n)` can be viewed as a subalgebra of `Mat(n+1)`,
representing the affine transformation `x -> l(x) + v` by: 
```python
(l, v) = [[l_11, ..., l_1n, v_1],
          [l_21, ..., l_2n, v_2],
           ...
          [l_n1, ..., l_nn, v_n],
          [   0, ...,    0,   1]]
```

This view will allow for inheritance of most matrix methods. 

override: `transpose`, restrict to the linear part.

implement:
```python
    to_linear   : (affine) -> linear
    to_vector   : (affine) -> vector
    apply       : (affine, vector) -> vector
```

### GeneralAffine

inherit: `exp` and `log`. 

### SpecialEuclidian

override: `inv`, transpose the linear part and revert translation. 


## Vector Spaces and Embedded Manifolds

Here the inheritance diagram with the different class
names is somehow still unclear. A picture attempt:
```python
                                                            
    Connection <---- LeviCivita                             
                                                            
                         '                                  
                                                           
                       Metric                                     
                                                 Vector      
                         '                         ^     
                         '                         '       
                                        .--- EmbeddedManifold
                     RiemannianMfd <---(                        
                                        `--- ProductRiemannianMfd
                                                   :        
                                                   v         
                                                Product            
```         

I think the picture might get simpler 
if metrics, connections, etc. were thought of as
auxiliary objects serving to provide a manifold 
with 'riemannian' methods. Something like:
```python
manifold.use_metric(metric)   
```
That way the methods a manifold exposes would depend on the structures 
it is equipped with. 

Overall, a first list of expected methods:

```python
# (riemannian) connection:
    exp : (vector, point1) -> point2
    log : (point2, point1) -> vector
# riemannian structure:
  * geodesic     : (point2, point1) -> (t -> point) 
  * dist         : (point2, point 1) -> scalar 
    inner_matrix : point -> matrix 
  * inner        : (vector1, vector2, point) -> scalar  
# embedding: 
    belongs      : point -> bool
    is_tangent   : vector -> bool
    to_tangent   : vector -> vector
```

( * ): `geodesic`, `inner-prod` and `dist` should be generically defined 
from other methods such as `exp`, `log`, and `inner-matrix` 
in the parent class for inheritance.

In general, `exp` and `log` may derive from a connection. 
In this case the `geodesic` terminology is somewhat confusing and could be aliased.

__Notes:__ (came up discussing with @nguigs)
+ the `EmbeddedManifold` interface should be shared by the previous Lie Groups, 
which somewhat raises the issue of the 2D-shape signature of their tangent vectors
and the metric tensor being 4D, 
together with its consequences on vectorization behaviour.  
    
    in the matrix case, instead of recasting tangent vectors to 1D by default, 
    interfacing with some properly einsum-vectorized methods such as 
    `bilinear: (tensor, vector, vector) -> scalar`, and 
    `add`, `span`... defined in each of the two parent classes 
    could be more convenient.

+ in the `EmbeddedManifold` class, the Levi-Civita connection 
can be numerically integrated
by orthogonal projections onto the the tangent space. 
This means that by default, `exp` and `log` may derive from `to_tangent` 
_if not overriden by closed formulas_.

    The `EmbdeddedManifold` class could then be used to represent 
    manifolds defined as level surfaces `{ f = cst }` 
    of smooth functions and their intersections, using autograd :scream:.  

    I suggest most of the ODE solving code be kept separate in a separate 
    `integrators.py` file by an import statement though.
    Hence the `EmbeddedManifold` class file would only define its objects interfaces
    (of which already defined manifolds would only inherit the metric tensor), 
    and this file could as well be imported to solve user-defined ODEs, 
    on the sphere for instance. 

+ As discussed with @ninamiolane
 and concerning Yann's polydisks, Alice's discretized curves 
and @nguigs's landmarks spaces: __product spaces__.  

    A clean `Manifold` class model being figured, many canonical methods could 
    be inherited by a `ProductManifold` child class. 
    I think this is also related to the first point 
    (matrices being 2D and vectorization) 
    products yield new object shapes and may not always 
    be represented by rectangular arrays, 
    thus raising the concern for a common interface reference. 
