from typing import Callable, List, Dict, Optional, Union
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace


class Metric(ABC):
    def __init__(self):
        self.manifold = None

    def setManifold(self, manifold):
        self.manifold = manifold

    @abstractmethod
    def exp(self, tangent_vec):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class Manifold(ABC):

    def __init__(self, metrics : List[Metric]):
        self.metrics = metrics
        for metric in metrics:
            metric.setManifold(self)

    def get_metrics(self) -> List[Metric]:
        return self.metrics
        
    
    def call_method_on_metrics(self, function_name, *args, **kwargs) -> Union[int, Dict[str, int]]:
        metrics = self.get_metrics()

        if len(metrics) > 1:
            res = {}
            for metric in metrics:
                res[metric.name()] = getattr(metric, function_name)(*args, **kwargs)
            return res
        elif len(metrics) == 1:
            return getattr(metrics[0], function_name)(*args, **kwargs)
        else :
            print(f"no metric to call function {function_name}")
            return 0

    
# if abstractManifoldFactory method create(self, metrics_names , **kwargs) and creator(kwargs)

# factory class needs to be a singleton for registration to work
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



class SpecialEuclidienManifoldFactory:
    """ Factory for Euclidien Manifold """
    metrics_creators = {}

    @classmethod
    def create(cls, *args, metrics_names : Optional[Union[str, List[str]]] = None, n: int = 2,  **kwargs):
        """ create a new manifold with associated metrics"""
        if not isinstance(metrics_names, list):
            print(f"{metrics_names} is a str, transforming to list")
            metrics_names = [metrics_names]

        metrics = cls._get_metrics(metrics_names)
        print(f"metrics created are {metrics}")
        if n == 2:
            return _SpecialEuclideanVec(metrics, *args, **kwargs)
        elif n == 3:
            return _SpecialEuclideanMatrices(metrics, *args, **kwargs)
        else:
            raise NotImplementedError

        
    @classmethod
    def register(cls, name: str) -> Callable:
        """ decorator to register a new class"""
        def inner_wrapper(wrapped_class: Metric) -> Callable:
            if name in cls.metrics_creators:
                print(f"Executor {name} already exists. Will replace it")
            cls.metrics_creators[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def metric_keys(cls):
        """ return list of metric keys for this class of manifold""" 
        return cls.metrics_creators.keys()

    @classmethod
    def _get_metrics(cls, metrics_name : List[str]) -> List[Metric]:
        res = []
        for m in metrics_name:
            metric = cls.metrics_creators[m]()
            print(f"creating metric of type {m} : {metric}")
            res.append(metric)

        return res


class _SpecialEuclideanMatrices(Manifold):
    def __init__(self, metrics : List[Metric], point_type : str = "matrix"):
        super().__init__(metrics)
        self.point_type = point_type

    def another_method(self):
        print("Matrices")



class _SpecialEuclideanVec(Manifold):
    def __init__(self, metrics : List[Metric], point_type : str = "matrix"):
        super().__init__(metrics)
        self.point_type = point_type

    def thats_my_method(self):
        print("vec")


@SpecialEuclidienManifoldFactory.register("LEFT")
class LeftEuclidianMetric(Metric):
    def exp(self, tangent_vec):
        print(f"call gauche witht manifold  {self.manifold}")
        return 10

    def name(self):
        return "LEFT"

@SpecialEuclidienManifoldFactory.register("RIGHT")
class RightEuclidianMetric(Metric):
    def exp(self, tangent_vec):
        print("call droite")
        return 20

    def name(self):
        return "RIGHT"


def main():
    factory = SpecialEuclidienManifoldFactory
    manifold = factory.create(n=2, point_type="matrix", metrics_names = ["LEFT", "RIGHT"])

    tangent_vec = "toto"

    res = manifold.call_method_on_metrics('exp', tangent_vec)
    for name, geo_points in res.items():
        print(name, geo_points)


    print("manifold2 (1 metric)")
    manifold2 = factory.create(n=2, point_type="matrix", metrics_names = "LEFT")
    res = manifold2.call_method_on_metrics('exp', tangent_vec)
    print(res)


if __name__ == '__main__':
    main()
    