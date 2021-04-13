from typing import Callable, List, Dict, Optional, Union
from abc import ABC, abstractmethod
import itertools

#import geomstats.backend as gs
#from geomstats.geometry.pre_shape import PreShapeSpace
import geomstats.visualization as visualization

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
    manifolds_creators = {}
    
    @classmethod
    def create(cls, metrics_names : Optional[Union[str, List[str]]] = None, **kwargs ):
        args_dict = kwargs
        
        # check the incremental combination of args to see if a key exist        
        for length in range(len(args_dict), 0, -1):
            for key in itertools.combinations(args_dict.items(), length):
                if key in cls.manifolds_creators:
                    print(f"found key {key}  from args {args_dict}")
                    
                    if not isinstance(metrics_names, list):
                        print(f"{metrics_names} is a str, transforming to list")
                        metrics_names = [metrics_names]

                    metrics = cls._get_metrics(metrics_names)
                    print(f"metrics created are {metrics}")
            
                    key_keys = [k for k,v in key]
                    rest_of_args =  {k:v for k,v in args_dict.items() if k not in key_keys}
                    return cls.manifolds_creators[key](metrics, rest_of_args)

        raise Exception(f"no manifold with key containing {args_dict}")
                
    @classmethod
    def register(cls, **kwargs):
        """decorator to register new manifold type

        Returns:
            Callable: [description]
        """
        def wrapper(manifold_class: Manifold):
            args_dict = kwargs
            key = tuple(sorted(args_dict.items()))  # TODO without sorted
            
            if key in cls.manifolds_creators:
                print(f"warning, this combination of args alreay exist: {key} . I will replace it")
            
            cls.manifolds_creators[key] = manifold_class
            return manifold_class 

        return wrapper
                        
    
    
    @classmethod
    def registerMetric(cls, name: str = None) -> Callable:
        """decorator to register a new Metric class

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

        

@SpecialEuclidienManifoldFactory.register(n=3)
class _SpecialEuclideanMatrices(Manifold):
    def __init__(self, metrics : List[Metric], point_type : str = "matrix"):
        super().__init__(metrics)
        self.point_type = point_type

    def another_method(self):
        print("Matrices")


@SpecialEuclidienManifoldFactory.register(n=2)
class _SpecialEuclideanVec(Manifold):
    def __init__(self, metrics : List[Metric], point_type : str = "matrix"):
        super().__init__(metrics)
        self.point_type = point_type

    def thats_my_method(self):
        print(self.point_type)


@SpecialEuclidienManifoldFactory.register(n=2, miaou="toto")
class _SpecialEuclideanDummy(Manifold):
    def __init__(self, metrics : List[Metric], point_type : str = "matrix", epsilon: float = 0.001):
        super().__init__(metrics)
        self.point_type = point_type
        self.epsilon = epsilon

    def thats_my_method(self):
        print(self.point_type)


@SpecialEuclidienManifoldFactory.registerMetric("LEFT")
class LeftEuclidianMetric(Metric):
    def exp(self, tangent_vec):
        print(f"call gauche witht manifold  {self.manifold}")
        return 10

    def name(self):
        return "LEFT"

@SpecialEuclidienManifoldFactory.registerMetric()
class RightEuclidianMetric(Metric):
    def exp(self, tangent_vec):
        print("call droite")
        return 20


def main(): 
    factory = SpecialEuclidienManifoldFactory
    
    print("First manifold")
    manifold = factory.create(n=3, point_type="matrix", metrics_names = ["LEFT", "RightEuclidianMetric"])

    tangent_vec = "toto"

    res = manifold.call_method_on_metrics('exp', tangent_vec)
    for name, geo_points in res.items():
        print(name, geo_points)

    print("\nSecond manifold")
    manifold2 = factory.create(n=2, point_type="miaou", metrics_names = "LEFT")
    res = manifold2.call_method_on_metrics('exp', tangent_vec)
    print(res)
    manifold2.thats_my_method()

    print("\nThird manifold")
    manifold3 = factory.create(miaou="toto", n=2, point_type="miaou", metrics_names = "LEFT", epsilon=0.002)
    res = manifold3.call_method_on_metrics('exp', tangent_vec)

    print("\nFourth manifold")
    try:
        manifold_dont_exist = factory.create(n=4, metrics_names = "LEFT")
    except:
        print("good, exception is here")


if __name__ == '__main__':
    main()
    