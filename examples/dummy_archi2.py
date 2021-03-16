from typing import List, Dict, Union
from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class Manifold(ABC):

    def __init__(self, metrics : List[Metric]):
        self.metrics = metrics

    def getMetrics(self) -> List[Metric]:
        return self.metrics

    @abstractmethod
    def exp(self, tangent_vec) -> Union[int, Dict[str, int]]:  # TODO return also only str 
        pass

    
# if abstractManifoldFactory method create(self, metrics_names , **kwargs) and creator(kwargs)

# factory class needs to be a singleton for registration to work
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



class SpecialEuclidienManifoldFactory(metaclass=SingletonMeta):
    def __init__(self):
        self.manifold_creators = {}
        self.metrics_creators = {}

    def create(self, n : int=2, point_type : str='matrix', epsilon : float =0. , metrics_names : Union[str, List[str]] = []):
        if not isinstance(metrics_names, list):
            print(f"{metrics_names} is a str, transforming to list")
            metrics_names = [metrics_names]

        metrics = self._getMetrics(metrics_names)
        print(f"metrics created are {metrics}")
        if n == 2:
            return _SpecialEuclideanVec(metrics)
        elif n == 3: 
            return _SpecialEuclideanMatrices(metrics)
        else:
            raise NotImplementedError

    def registerMetric(self, name: str, metric_creator):
        self.metrics_creators[name] = metric_creator

    def metricKeys(self) : 
        return self.metrics_creators.keys()

    def _getMetrics(self, metrics_name : List[str]) -> List[Metric]:
        res = []
        for m in metrics_name:
            metric = self.metrics_creators[m]()
            print(f"creating metric of type {m} : {metric}")
            res.append(metric)

        return res


class _SpecialEuclideanMatrices(Manifold):
    def exp(self, tangent_vec)  -> Union[int, Dict[str, int]]:
        print("Matrices")
        metrics = super.getMetrics()

        if len(metrics) > 1:
            res = {}
            for metric in super().getMetrics():
                res[metric.name()] = metric() # TODO tangent_vec
            return res
        elif len(metrics) == 1:
            return metrics[0]()
        else :
            print("no metric")
            return 0


class _SpecialEuclideanVec(Manifold):
    def exp(self, tangent_vec)  -> Union[int, Dict[str, int]]:
        print("Vec")
        metrics = super().getMetrics()

        if len(metrics) > 1:
            res = {}
            for metric in super().getMetrics():
                res[metric.name()] = metric() # TODO tangent_vec
            return res
        elif len(metrics) == 1:
            return metrics[0]()
        else :
            print("no metric")
            return 0


class LeftEuclidianMetric(Metric):
    def __call__(self):
        print("call gauche")
        return 10

    def name(self):
        return "LEFT"    

# to be done by a decorator
SpecialEuclidienManifoldFactory().registerMetric("LEFT", LeftEuclidianMetric)

class RightEuclidianMetric(Metric):
    def __call__(self):
        print("call droite")
        return 20

    def name(self):
        return "RIGHT"   

# to be done by a decorator
SpecialEuclidienManifoldFactory().registerMetric("RIGHT", RightEuclidianMetric)



def main():
    factory = SpecialEuclidienManifoldFactory()
    manifold = factory.create(n=2, point_type="matrix", metrics_names = ["LEFT", "RIGHT"])

    tangent_vec = "toto"

    res = manifold.exp(tangent_vec)
    for name, geo_points in res.items():
        print(name, geo_points)


    print("manifold2 (1 metric")
    manifold2 = factory.create(n=2, point_type="matrix", metrics_names = "LEFT")
    res = manifold2.exp(tangent_vec)
    print(res)



if __name__ == '__main__':
    main()