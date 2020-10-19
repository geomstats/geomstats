"""The kernel density estimation classifier on manifolds."""

import math

from sklearn.neighbors import RadiusNeighborsClassifier

import geomstats.backend as gs


class KernelDensityEstimationClassifier(RadiusNeighborsClassifier):
    """Classifier implementing the kernel density estimation on manifolds.

    The kernel density estimation classifier classifies the data according to
    a kernel density estimation of each dataset on the manifold. The density
    estimation is performed using radial kernel functions: the distance
    is the only geometrical tool used to estimate the density on the manifold.
    This classifier inherits from the radius neighbors classifier of the
    scikit-learn library, we expect the classifier presented here to be easier
    to use on manifolds.
    Compared with the radius neighbors classifier, we force the
    parameter 'algorithm' to be equal to 'brute' in order to
    be compatible with any metric.
    We also changed some default values of the scikit-learn algorithm in order
    to take into account every point of the dataset during the kernel density
    estimation, i.e. the default value of the parameter 'radius' is set to
    infinity instead of 1 and the default value of the parameter 'weight' is
    set to 'distance' instead of 'uniform'.
    Our main contribution is a greater choice of kernel functions,
    see the radial_kernel_functions.py file in the learning directory.
    The radial kernel functions are now easier to define by a user:
    the input data should be an array of distances instead of an array of
    arrays. Moreover the new parameter 'bandwidth' of our classifier can be
    used to adapt the kernel function to the size of the dataset.
    The scikit-learn library also provides a kernel density estimation tool
    (see sklearn.neighbors.KernelDensity), however this algorithm is not built
    as a classifier and is not available with all metrics.

    Parameters
    ----------
    radius : float, optional (default = inf)
        Range of parameter space to use by default.
    kernel : string or callable, optional (default = 'distance')
        Kernel function used in prediction. Possible values:
        - 'distance' : weight points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    bandwidth : float, optional (default = 1.0)
        Bandwidth parameter used for the kernel. The kernel parameter is
        used if and only if the kernel is a callable function.
    p : integer, optional (default = 2)
        Power parameter for the 'minkowski' string distance.
        When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used.
    distance : string or callable, optional (default = 'minkowski')
        The distance metric to use.
        The default distance is minkowski, and with p=2 is equivalent to the
        standard Euclidean distance.
        See the documentation of the DistanceMetric class in the scikit-learn
        library for a list of available distances.
        If distance is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
    outlier_label : {manual label, 'most_frequent'}, optional (default = None)
        Label for outlier samples (samples with no neighbors in given radius).
        - manual label: str or int label (should be the same type as y)
          or list of manual labels if multi-output is used.
        - 'most_frequent' : assign the most frequent label of y to outliers.
        - None : when any outlier is detected, ValueError will be raised.
    distance_params : dict, optional (default = None)
        Additional keyword arguments for the distance function.
    n_jobs : int or None, optional (default = None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1; ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : array-like, shape=[n_classes,]
        Class labels known to the classifier.
    effective_metric_ : string or callable
        The distance metric used. It will be same as the `distance` parameter
        or a synonym of it, e.g. 'euclidean' if the `distance` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the distance function.
        For most distances will be same with `distance_params` parameter,
        but may also contain the `p` parameter value if the
        `effective_metric_` attribute is set to 'minkowski'.
    outputs_2d_ : bool
        False when `y`'s shape is [...,] or [..., 1] during fit,
        otherwise True.

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://scikit-learn.org/stable/modules/generated/
    sklearn.neighbors.RadiusNeighborsClassifier.html
    """

    def __init__(self, radius=math.inf,
                 kernel='distance',
                 bandwidth=1.0,
                 p=2,
                 distance='minkowski',
                 distance_params=None,
                 n_jobs=None,
                 **kwargs):

        self.bandwidth = bandwidth

        if isinstance(kernel, str):
            weights = kernel
        else:
            def weights(distance_matrix):
                n_samples = distance_matrix.shape[0]
                weights_list = []
                for i_sample in range(n_samples):
                    weights_list.append(
                        kernel(
                            distance=distance_matrix[i_sample],
                            bandwidth=self.bandwidth))
                weights_matrix = gs.array(weights_list)
                return weights_matrix

        super().__init__(
            radius=radius,
            weights=weights,
            algorithm='brute',
            p=p,
            metric=distance,
            metric_params=distance_params,
            n_jobs=n_jobs,
            **kwargs)

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : array-like, shape=[..., n_features], [...,] or
            [..., n_samples] if distance is 'precomputed'
            Training data.
        y : array-like, shape=[...,] or [..., n_outputs]
            Target values.
        """
        data_shape = gs.shape(X)
        if len(data_shape) == 1:
            n_samples = data_shape[0]
            X = gs.reshape(X, (n_samples, 1))
        super(KernelDensityEstimationClassifier, self).fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape=[n_queries, n_features] or
            [n_queries, n_indexed] if metric is 'precomputed'
            Test samples.

        Returns
        -------
        y : array-like, shape=[n_queries] or [n_queries, n_outputs]
            Class labels for each data sample.
        """
        data_shape = gs.shape(X)
        if len(data_shape) == 1:
            n_samples = data_shape[0]
            X = gs.reshape(X, (n_samples, 1))
        y_pred = super(KernelDensityEstimationClassifier, self).predict(X)
        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape=[n_queries, n_features] or
            [n_queries, n_indexed] if metric is 'precomputed'
            Test samples.

        Returns
        -------
        probabilities : array-like, shape=[n_queries, n_classes] or a list of
            n_outputs of such arrays if n_outputs > 1
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        data_shape = gs.shape(X)
        if len(data_shape) == 1:
            n_samples = data_shape[0]
            X = gs.reshape(X, (n_samples, 1))
        probabilities = super(
            KernelDensityEstimationClassifier, self).predict_proba(X)
        return probabilities
