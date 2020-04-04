"""The MDM classifier on manifolds."""

from geomstats.learning.frechet_mean import FrechetMean
import numpy

class RiemannianMinimumDistanceToMeanClassifier():
    '''
    Classifier implementing the MDM scheme on manifolds.

    Parameters
    ----------
    metric : string or callable, optional (default = 'minkowski')
        The distance metric to use.
        The default metric is minkowski, and with p=2 is equivalent to the
        standard Euclidean metric.
        See the documentation of the DistanceMetric class in the scikit-learn
        library for a list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.

    Attributes
    ----------
    G: array-like, shape=[n_classes, n_features] if point_type='vector'
                   shape=[n_classes, n_features, n_features] if point_type='matrix'
       Frechet means of each class.
    '''

    def __init__(self, riemannian_metric,mean_method='default',verbose=0,point_type='vector'):
        self.riemannian_metric=riemannian_metric
        self.point_type=point_type

    def fit(self,X,Y):
        '''
        Compute Frechet mean of each class.
        :param X: array-like, shape=[n_samples, n_features] if point_type='vector'
                              shape=[n_samples, n_features, n_features] if point_type='matrix'
                  Training data, where n_samples is the number of samples and n_features is the number of features.
        :param Y: array-like, shape=[n_samples, n_classes]
                  Training labels, where n_classes is the number of classes.
        # :return: G: array-like, shape=[n_classes, n_features] if point_type='vector'
        #                         shape=[n_classes, n_features, n_features] if point_type='matrix'
        #             Frechet means of each class.
        '''
        n_classes=Y.shape[-1]
        mean_estimator=FrechetMean(metric=self.riemannian_metric,point_type=self.point_type)
        frechet_means=[]
        for c in range(n_classes):
            data_class=self.split_data_in_classes(X,Y,c)
            # data_class_vec=numpy.array([sample_data.mat2vec(data_class[i]) for i in range(n_samples)])
            # data_class_vec=data_class.reshape((-1,n*n))
            frechet_means.append(mean_estimator.fit(data_class).estimate_[0])
        self.G=numpy.array(frechet_means)
        return

    def predict(self,X):
        '''
        Compute closest neighbor according to riemannian_metric
        :param X: array-like, shape=[n_samples, n_features] if point_type='vector'
                              shape=[n_samples, n_features, n_features] if point_type='matrix'
                  Test data, where n_samples is the number of samples and n_features is the number of features.
        :return: Y: array-like, shape=[n_samples, n_classes]
                    Predicted labels, where n_classes is the number of classes.
        '''
        n_samples=X.shape[0]
        n_classes=self.G.shape[0]
        Y=numpy.zeros((n_samples,n_classes))
        for i in range(n_samples):
            c=self.riemannian_metric.closest_neighbor_index(X[i],self.G)
            Y[i,c]=1
        return Y

    def split_data_in_classes(self,X,Y,c):
        '''
        Splits a labelled dataset in sub-datasets of each label
        :param X: array-like, shape=[n_samples, n_features] if point_type='vector'
                              shape=[n_samples, n_features, n_features] if point_type='matrix'
                  Labelled dataset, where n_samples is the number of samples and n_features is the number of features.
        :param Y: array-like, shape=[n_samples, n_classes]
                  Labels, where n_classes is the number of classes.
        :param c: int
                  Class index
        :return: array-like, shape=[n_samples_in_class, n_features] if point_type='vector'
                              shape=[n_samples_in_class, n_features, n_features] if point_type='matrix'
                  Labelled dataset, where n_samples_in_class is the number of samples in class c
        '''
        return X[numpy.where(numpy.where(Y)[1]==c)]