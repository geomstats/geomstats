"""Illustration for the MDM classification of SPD matrices in  dim 2."""


import geomstats.datasets.sample_sdp_2d
import geomstats.visualization as visualization
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.spd_matrices import SPDMetricAffine
from geomstats.learning.mdm import RiemannianMinimumDistanceToMeanClassifier

def main():
    """Execute illustration of MDM supervised classifier."""
    n_samples = 100
    n_features = 2
    n_classes = 3

    # generate toy dataset of 2D SPD matrices
    dataset_generator = geomstats.datasets.sample_sdp_2d.DatasetSPD2D(
        n_samples, n_features, n_classes)
    data, labels = dataset_generator.generate_sample_dataset()

    # plot dataset as ellipses
    ellipsis = visualization.Ellipsis2D()
    for i in range(n_samples):
        x = data[i]
        y = dataset_generator.data_helper.get_label_at_index(i, labels)
        ellipsis.draw(x, color=ellipsis.colors[y], alpha=.1)

    # define and fit MDM classifier to data
    metric = SPDMetricAffine(n=n_features)
    MDMEstimator = RiemannianMinimumDistanceToMeanClassifier(
        metric, n_classes, point_type='matrix')
    MDMEstimator.fit(data, labels)

    # plot Frechet means computed in the MDM
    for i in range(n_classes):
        ellipsis.draw(
            MDMEstimator.mean_estimate[i],
            color=ellipsis.colors_alt[i],
            linewidth=5,
            label='Barycenter of class ' + str(i))

    # generate random test samples, and predict with MDM classifier
    data_test = SPDMatrices(n=n_features).random_uniform(n_samples=3)
    predictions = MDMEstimator.predict(data_test)

    for i in range(data_test.shape[0]):
        c = list(predictions[i] == 1).index(True)
        x_from, y_from = ellipsis.draw(
            data_test[i], color=ellipsis.colors[c], linewidth=5)
        _, _, x_to, y_to = ellipsis.compute_coordinates(
            MDMEstimator.mean_estimate[c])
        arrow = visualization.DataArrow(ellipsis.fig)
        arrow.draw(x_from, y_from, x_to, y_to)

    ellipsis.plot()


if __name__ == '__main__':
    main()
