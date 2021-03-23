from typing import Callable, List, Dict, Optional, Union
from abc import ABC, abstractmethod

#import geomstats.backend as gs
#from geomstats.geometry.pre_shape import PreShapeSpace


class Metric(ABC):
    def __init__(self):
        self.manifold = None

    def setManifold(self, manifold):
        self.manifold = manifold

    @abstractmethod
    def exp(self, tangent_vec):
        pass

    @classmethod
    def name(cls) -> str:
        return cls.__name__


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

    

class AbstractManifoldFactory(ABC):
    metrics_creators = {}
    
    @classmethod
    @abstractmethod
    def create(cls, *args, metrics_names : Optional[Union[str, List[str]]] = None, **kwargs ):
        pass
    
    @classmethod
    def register(cls, name: str = None) -> Callable:
        """decorator to register a new class

        Args:
            name (str): the name of the metric to Register 
            
        Returns:
            Callable: a metric creator
        """
        def inner_wrapper(wrapped_class: Metric) -> Callable:
            inner_name = name
            if inner_name is None:
                print(f"no name, will use class name {wrapped_class.__name__}")
                inner_name = wrapped_class.__name__
                
            if inner_name in cls.metrics_creators:
                print(f"Executor {inner_name} already exists. Will replace it")
            cls.metrics_creators[inner_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def metric_keys(cls):
        """getter for list of metric keys

        Returns:
            List[str]: a list of metric keys for this class of manifold
        """
         
        return cls.metrics_creators.keys()

    @classmethod
    def _get_metrics(cls, metrics_name : List[str]) -> List[Metric]:
        """internal method to create metrics from a list of names

        Args:
            metrics_name (List[str]): List of metrics names 

        Returns:
            List[Metric]: List of metrics
        """
        
        res = []
        for m in metrics_name:
            if m not in cls.metrics_creators:
                print(f"error {m} not in metrics keys: {cls.metrics_creators.keys()}")
                continue 
            metric = cls.metrics_creators[m]()
            res.append(metric)

        return res
        #return [cls.metrics_creators[m] for m in metrics_name if m in cls.metrics_creators]

    


class SpecialEuclidienManifoldFactory(AbstractManifoldFactory):
    """ Factory for Euclidien Manifold """

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

@SpecialEuclidienManifoldFactory.register()
class RightEuclidianMetric(Metric):
    def exp(self, tangent_vec):
        print("call droite")
        return 20

    #def name(self):
    #    return "RIGHT"


def main(): 
    factory = SpecialEuclidienManifoldFactory
    manifold = factory.create(n=2, point_type="matrix", metrics_names = ["LEFT", "RightEuclidianMetric"])

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
    