import random

import geomstats.backend as gs

class K_Means(BaseEstimator, ClusterMixin):

    def __init__(self, metric, n_component, seeds=None, n_jobs=None):
        self.metric = metric
        self.means = gs.rand(n_component, metric.dimension)



    def fit(self, X, Y=None, max_iter=1000):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : If given only give the metric for each labels

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 
        to_component = gs.zeros(X.shape[0])
        self.centroids = gs.vstack([gs[random.randint(0, to_component-1)] 
                                    for i in range(to_component)])
        index = 0 

        while(index < max_iter):
            index+=1
            # expectation 
            metric.dist()
            # maximisation
            [self.metric.mean(0) for i in range(self.n_component)]            

            
    def predict(self, X):
        

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict()


class PoincareKMeansNInit(object):
    def __init__(self, n_clusters, min_cluster_size=5, verbose=False, init_method="random", n_init=20):
        self.verbose = verbose
        self.KMeans = [PoincareKMeans(n_clusters, min_cluster_size, verbose, init_method) for i in range(n_init)]
    
    def fit(self, X, Y=None, max_iter=10):
        pb = tqdm.trange(len(self.KMeans))
        stds = torch.zeros(len(self.KMeans))
        # print("X.device : ",X.device)
        for i, kmeans in zip(pb,self.KMeans):
            kmeans.fit(X, Y, max_iter)
            stds[i] = kmeans.getStd(X).mean()
        self.min_std_val, self.min_std_index = stds.min(-1)
        self.min_std_val, self.min_std_index = self.min_std_val.item(), self.min_std_index.item()
        self.kmean = self.KMeans[self.min_std_index]
        self.centroids = self.kmean.centroids
        self.cluster_centers_  =  self.centroids 

    def predict(self, X):
        return self.kmean._expectation(self.centroids, X)

    def getStd(self, X):
        return self.kmean.getStd(X)

class PoincareKMeans(object):
    def __init__(self, n_clusters, min_cluster_size=2, verbose=False, init_method="random"):
        self._n_c = n_clusters
        self._distance = pf.distance
        self.centroids = None
        self._mec = min_cluster_size
        self._init_method = init_method

    def _maximisation(self, x, indexes):
        centroids = x.new(self._n_c, x.size(-1))
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= self._mec):
                lx = x[random.randint(0,len(x)-1)].unsqueeze(0)
            centroids[i] = pa.barycenter(lx, normed=True)
        return centroids
    
    def _expectation(self, centroids, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)
        value, indexes = dst.min(-1)
        return indexes

    def _init_random(self, X):
        self.centroids_index = (torch.rand(self._n_c, device=X.device) * len(X)).long()
        self.centroids = X[self.centroids_index]

    def __init_kmeansPP(self, X):
        distribution = torch.ones(len(X))/len(X)
        frequency = pytorch_categorical.Categorical(distribution)
        centroids_index = []
        N, D = X.shape
        while(len(centroids_index)!=self._n_c):

            f = frequency.sample(sample_shape=(1,1)).item()
            if(f not in centroids_index):
                centroids_index.append(f)
                centroids = X[centroids_index]
                x = X.unsqueeze(1).expand(N, len(centroids_index), D)
                dst = self._distance(centroids, x)
                value, indexes = dst.min(-1)
                vs = value**2
                distribution = vs/(vs.sum())
                frequency = pytorch_categorical.Categorical(distribution)
        self.centroids_index = torch.tensor(centroids_index, device=X.device).long()
        self.centroids = X[self.centroids_index]

    def fit(self, X, Y=None, max_iter=100):
        lt = []
        ft = 0
        if(Y is None):
            with torch.no_grad():
                if(self._mec < 0):
                    self._mec = len(X)/(self._n_c**2)
                if(self.centroids is None):
                    if(self._init_method == "kmeans++"):
                        self.__init_kmeansPP(X)
                    else:
                        self._init_random(X)
                for iteration in range(max_iter):

                    if(iteration >= 1):
                        old_indexes = self.indexes
                    start_time = time.time()
                    self.indexes = self._expectation(self.centroids, X)
                    self.centroids = self._maximisation(X, self.indexes)
                    end_time = time.time()


                    if(iteration >= 1):   
                        if((old_indexes == self.indexes).float().mean() == 1):
                            # print(" Iteration end : ", iteration)
                            self.cluster_centers_  =  self.centroids
                            # print("first ", lt[0])
                            # print(lt)
                            # print("time mean ", sum(lt,0)/iteration)
                            # print("NB iter ", iteration)
                            return self.centroids
                self.cluster_centers_  =  self.centroids
                return self.centroids
        else:
            self.indexes = Y.max(-1)[1]
            self.centroids = self._maximisation(X, self.indexes)
            self.cluster_centers_  =  self.centroids
            # print(self.centroids)
            return self.centroids

    def predict(self, X):
        return self._expectation(self.centroids, X)

    def getStd(self, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = self.centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)**2
        value, indexes = dst.min(-1)
        stds = []
        for i in range(self._n_c):
            stds.append(value[indexes==i].sum())
        stds = torch.Tensor(stds)
        return stds
    def probs(self, X):
        predicted = self._expectation(self.centroids, X).squeeze().tolist()
        res = torch.zeros(len(X), self._n_c)
        for i, l in enumerate(predicted):
            res[i][l] = 1
        return res