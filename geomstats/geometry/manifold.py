"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.
"""

import itertools
import logging
from abc import ABC
from typing import Callable, Dict, List, Optional, Union

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.connection import Connection


class Manifold(ABC):
    r"""Class for manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    default_point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: 'vector'.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
            self, dim,
            metrics: List[Connection] = None,
            default_point_type='vector',
            default_coords_type='intrinsic', **kwargs):
        super(Manifold, self).__init__()
        geomstats.errors.check_integer(dim, 'dim')
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, 'default_point_type', ['vector', 'matrix'])

        self.dim = dim
        self.default_point_type = default_point_type
        self.default_coords_type = default_coords_type
        self._metrics = metrics or []
        for metric in self._metrics:
            metric.setManifold(self)

    def belongs(self, point, atol=gs.atol):
        """
        Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        raise NotImplementedError('belongs is not implemented.')

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError(
            'is_tangent is not implemented.')

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        raise NotImplementedError(
            'to_tangent is not implemented.')

    def random_point(self, n_samples=1, bound=1.):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        raise NotImplementedError('random_point is not implemented')

    def regularize(self, point):
        """Regularize a point to the canonical representation for the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., dim]
            Regularized point.
        """
        regularized_point = point
        return regularized_point

    def getMetrics(self) -> List[Connection]:
        """Get the list of metrics associated with this manifold.

        Returns
        -------
        A list of metrics
        """
        return self._metrics

    def call_method_on_metrics(self,
                               function_name: str,
                               *args,
                               **kwargs) -> Union[any, Dict[str, any]]:
        """Call a method on all metrics of this manifold.

        Parameters
        ----------
            function_name (str): method name to call on metrics

        Returns
        -------
            Union[any, Dict[str, any]]: either the result or a dict
              containing metrics names and their associated result
        """
        metrics = self.getMetrics()

        if len(metrics) > 1:
            res = {}
            for metric in metrics:
                res[metric.name()] = getattr(metric,
                                             function_name)(*args, **kwargs)
            return res

        if len(metrics) == 1:
            return getattr(metrics[0], function_name)(*args, **kwargs)

        logging.warning(f"no metric to call function {function_name}")
        return None


class AbstractManifoldFactory(ABC):
    """Abstract class to easily create Manifold factories."""

    metrics_creators = {}
    manifolds_creators = {}

    @classmethod
    def create(cls,
               metrics_names: Optional[Union[str, List[str]]] = None,
               **kwargs):
        """Create a manifold with it's metrics.

        Returns
        -------
            Manifold: the new manifold
        """
        args_dict = kwargs

        # check the incremental combination of args to see if a key exist
        nb_args = len(args_dict)

        for length in range(nb_args, 0, -1):
            logging.debug(f" test with key length {length}")
            for key in itertools.combinations(sorted(args_dict.items()),
                                              length):
                if key in cls.manifolds_creators:
                    logging.debug(f"found key {key}  from args {args_dict}")

                    if metrics_names is not None:
                        if not isinstance(metrics_names, list):
                            logging.debug(f'''
                                          {metrics_names} is a str,
                                          transforming to list''')
                            metrics_names = [metrics_names]

                        metrics = cls._get_metrics(metrics_names)
                        logging.debug(f"metrics created are {metrics}")
                    else:
                        metrics = None

                    key_keys = [k for k, v in key]
                    rest_of_args = {k: v for k, v in args_dict.items() if k not in key_keys} # NOQA
                    if 'dim' in key_keys:
                        rest_of_args['dim'] = args_dict['dim']

                    return cls.manifolds_creators[key](metrics=metrics, **rest_of_args) # NOQA

        raise Exception(f'''no manifold with key containing '''
                        f'''{sorted(args_dict.items())} .'''
                        f'''keys ars {cls.manifolds_creators.keys()}''')

    @classmethod
    def register(cls, **kwargs):
        """Register a new manifold class.

        Returns
        -------
            Callable: [description]
        """
        def wrapper(manifold_class: Manifold): # NOQA
            args_dict = kwargs
            key = tuple(sorted(args_dict.items()))

            if key in cls.manifolds_creators:
                logging.info(f'''for manifold {cls} this combination'''
                             f''' of args alreay exist: {key} .'''
                             '''I will replace it''')

            cls.manifolds_creators[key] = manifold_class
            return manifold_class

        return wrapper

    @classmethod
    def registerMetric(cls, name: str = None) -> Callable:
        """Register a new metric class.

        Parameters
        ----------
            name (str): the name of the metric to Register

        Returns
        -------
            Callable: a metric creator
        """
        def inner_wrapper(wrapped_class: Connection) -> Callable: # NOQA
            inner_name = name
            if inner_name is None:
                logging.debug('''register new metric without '''
                              '''name, will use class name'''
                              f''' {wrapped_class.__name__}''')
                inner_name = wrapped_class.__name__

            if inner_name in cls.metrics_creators:
                logging.info('''Metric creator with key'''
                             f'''{inner_name} already exists.'''
                             ''' I Will replace it''')
            cls.metrics_creators[inner_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def metric_keys(cls):
        """Getter for list of metric keys.

        Returns
        -------
            List[str]: a list of metric keys for this class of manifold
        """
        return cls.metrics_creators.keys()

    @classmethod
    def _get_metrics(cls, metrics_name: List[str]) -> List[Connection]:
        """Create create metrics from a list of names.

        Parmeters
        ---------
            metrics_name (List[str]): List of metrics names

        Returns
        -------
            List[Connection]: List of metrics
        """
        res = []
        for m in metrics_name:
            if m not in cls.metrics_creators:
                logging.warning(f'''{m} not in metrics '''
                                f'''keys: {cls.metrics_creators.keys()}''')
                continue
            metric = cls.metrics_creators[m]()
            res.append(metric)

        return res
