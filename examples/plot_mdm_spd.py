from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.mdm import RiemannianMinimumDistanceToMeanClassifier

import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import plot_mdm_spd_helper

def main():

    n_samples=100
    n_features=2
    n_classes=3

    # generate toy dataset of 2D SPD matrices
    dataset_generator=plot_mdm_spd_helper.DatasetSPD_2D(n_samples,n_features,n_classes)
    data,labels=dataset_generator.generate_sample_dataset()

    # plot dataset as ellipses
    plot_helper=plot_mdm_spd_helper.PlotHelper()
    for i in range(n_samples):
        x=data[i]
        y=dataset_generator.data_helper.get_label_at_index(i,labels)
        plot_helper.plot_ellipse(x,color=plot_helper.colors[y],alpha=.1)

    # define and fit MDM classifier to data
    metric=SPDMetricAffine(n=n_features)
    MDMEstimator=RiemannianMinimumDistanceToMeanClassifier(metric,point_type='matrix')
    MDMEstimator.fit(data,labels)

    # plot Frechet means computed in the MDM
    # for i in range(n_classes):
    #     plot_helper.plot_ellipse(MDMEstimator.G[i],color=plot_helper.colors_alt[i],linewidth=5)

    # data_vec=numpy.array([sample_data.mat2vec(data[i]) for i in range(n_samples)])
    # data_vec=numpy.reshape(data,(-1,n*n))

    # generate random test samples, and predict with MDM classifier
    data_test=SPDMatrices(n=n_features).random_uniform(n_samples=3)
    predictions=MDMEstimator.predict(data_test)

    for i in range(data_test.shape[0]):
        plot_helper.plot_ellipse(data_test[i],color=plot_helper.colors[i],linewidth=5)

    plt.show()

    # clustering = RiemannianKMeans(riemannian_metric=SPDMetricAffine(n=n),n_clusters=n_classes,point_type='matrix')
    # clustering = clustering.fit(data,max_iter=10)

if(__name__=='__main__'):
    main()