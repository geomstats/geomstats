"""Computations on the Lie group of rigid transformations."""

import logging
import numpy as np

import geomstats.rotations as rotations

TRANSLATIONS_IDENTITY = np.array([0, 0, 0])

GROUP_IDENTITY = np.concatenate([rotations.GROUP_IDENTITY,
                                 TRANSLATIONS_IDENTITY])
ALGEBRA_CANONICAL_INNER_PRODUCT = np.eye(6)


def regularize_transformation(transfo):
    """
    Regularize an element of the group SE(3),
    by extracting the rotation vector r from the input [r t]
    and using rotations.regularize_rotation_vector.

    :param transfo: 6d vector, element in SE(3) represented as [r t].
    :returns regularized_transfo: 6d vector, element in SE(3)
    with regularized rotation.
    """

    regularized_transfo = transfo
    rot_vec = transfo[0:3]
    regularized_transfo[0:3] = rotations.regularize_rotation_vector(rot_vec)

    return regularized_transfo


def inverse(transfo):
    """
    Compute the group inverse in SE(3).

    Formula:
    (R, t)^{-1} = (R^{-1}, R^{-1}.(-t))

    :param transfo: 6d vector element in SE(3)
    :returns inverse_transfo: 6d vector inverse of transfo
    """

    transfo = regularize_transformation(transfo)

    rot_vec = transfo[0:3]
    translation = transfo[3:6]

    inverse_transfo = np.zeros(6)
    inverse_transfo[0:3] = -rot_vec
    rot_mat = rotations.rotation_matrix_from_rotation_vector(-rot_vec)
    inverse_transfo[3:6] = np.dot(rot_mat, -translation)

    inverse_transfo = regularize_transformation(inverse_transfo)
    return inverse_transfo


def compose(transfo_1, transfo_2):
    """
    Compose two elements of group SE(3).

    Formula:
    transfo_1 . transfo_2 = [R1 * R2, (R1 * t2) + t1]
    where:
    R1, R2 are rotation matrices,
    t1, t2 are translation vectors.

    :param transfo_1, transfo_2: 6d vectors elements of SE(3)
    :returns prod_transfo: composition of transfo_1 and transfo_2
    """
    rot_mat_1 = rotations.rotation_matrix_from_rotation_vector(transfo_1[0:3])
    rot_mat_1 = rotations.closest_rotation_matrix(rot_mat_1)

    rot_mat_2 = rotations.rotation_matrix_from_rotation_vector(transfo_2[0:3])
    rot_mat_2 = rotations.closest_rotation_matrix(rot_mat_2)

    translation_1 = transfo_1[3:6]
    translation_2 = transfo_2[3:6]

    prod_transfo = np.zeros(6)
    prod_rot_mat = np.dot(rot_mat_1, rot_mat_2)

    prod_transfo[0:3] = rotations.rotation_vector_from_rotation_matrix(
            prod_rot_mat)
    prod_transfo[3:6] = np.dot(rot_mat_1, translation_2) + translation_1

    prod_transfo = regularize_transformation(prod_transfo)
    return prod_transfo


def jacobian_translation(transfo, left_or_right='left'):
    """
    Compute the jacobian matrix of the differential
    of the left/right translations
    from the identity to transfo in the Lie group SE(3).

    :param transfo: 6D vector element of SE(3)
    :returns jacobian: 6x6 matrix
    """
    assert len(transfo) == 6
    assert left_or_right in ('left', 'right')

    transfo = regularize_transformation(transfo)

    rot_vec = transfo[0:3]
    translation = transfo[3:6]

    jacobian = np.zeros((6, 6))

    if left_or_right == 'left':
        jacobian_rot = rotations.jacobian_translation(rot_vec,
                                                      left_or_right='left')
        jacobian_trans = rotations.rotation_matrix_from_rotation_vector(
                rot_vec)

        jacobian[:3, :3] = jacobian_rot
        jacobian[3:, 3:] = jacobian_trans

    else:
        jacobian_rot = rotations.jacobian_translation(rot_vec,
                                                      left_or_right='right')
        jacobian[:3, :3] = jacobian_rot
        jacobian[3:, :3] = -rotations.skew_matrix_from_vector(translation)
        jacobian[3:, 3:] = np.eye(3)

    return jacobian


def group_exp(tangent_vector,
              ref_point=GROUP_IDENTITY,
              epsilon=1e-5):
    """
    Compute the group exponential of vector tangent_vector,
    at point ref_point.

    :param tangent_vector: tangent vector of SE(3) at ref_point.
    :param ref_point: 6d vector element of SE(3).
    :returns group_exp_transfo: 6d vector element of SE(3).
    """
    tangent_vector = regularize_transformation(tangent_vector)

    if ref_point is GROUP_IDENTITY:
        rot_vec = tangent_vector[0:3]
        translation = tangent_vector[3:6]  # this is dt
        angle = np.linalg.norm(rot_vec)

        group_exp_transfo = np.zeros(6)
        group_exp_transfo[0:3] = rot_vec

        skew_mat = rotations.skew_matrix_from_vector(rot_vec)

        if angle == 0:
            coef_1 = 0
            coef_2 = 0
        elif angle < epsilon:
            coef_1 = 1. / 2. - angle ** 2 / 12.
            coef_2 = 1. - angle ** 3 / 3.
        else:
            coef_1 = (1. - np.cos(angle)) / angle ** 2
            coef_2 = (angle - np.sin(angle)) / angle ** 3

        sq_skew_mat = np.dot(skew_mat, skew_mat)
        group_exp_transfo[3:6] = (translation
                                  + coef_1 * np.dot(skew_mat, translation)
                                  + coef_2 * np.dot(sq_skew_mat, translation))

    else:
        ref_point = regularize_transformation(ref_point)

        jacobian = jacobian_translation(ref_point, left_or_right='left')
        inv_jacobian = np.linalg.inv(jacobian)

        tangent_vector_at_identity = np.dot(inv_jacobian, tangent_vector)
        group_exp_from_identity = group_exp(tangent_vector_at_identity)

        group_exp_transfo = compose(ref_point, group_exp_from_identity)

    return regularize_transformation(group_exp_transfo)


def group_log(transfo,
              ref_point=GROUP_IDENTITY,
              epsilon=1e-5):
    """
    Compute the group logarithm of point transfo,
    from point ref_point.

    :param transfo: 6d tangent_vector element of SE(3)
    :param ref_point: 6d tangent_vector element of SE(3)

    :returns tangent vector: 6d tangent vector at ref_point.
    """

    transfo = regularize_transformation(transfo)
    if ref_point is GROUP_IDENTITY:
        rot_vec = transfo[0:3]
        angle = np.linalg.norm(rot_vec)
        translation = transfo[3:6]

        tangent_vector = np.zeros(6)
        tangent_vector[0:3] = rot_vec
        skew_rot_vec = rotations.skew_matrix_from_vector(rot_vec)
        sq_skew_rot_vec = np.dot(skew_rot_vec, skew_rot_vec)

        if angle == 0:
            coef_1 = 0
            coef_2 = 0
            tangent_vector[3:6] = translation

        elif angle < epsilon:
            coef_1 = - 0.5
            coef_2 = 0.5 - angle ** 2 / 90

        else:
            coef_1 = - 0.5
            psi = 0.5 * angle * np.sin(angle) / (1 - np.cos(angle))
            coef_2 = (1 - psi) / (angle ** 2)

        tangent_vector[3:6] = (translation
                               + coef_1 * np.dot(skew_rot_vec, translation)
                               + coef_2 * np.dot(sq_skew_rot_vec,
                                                 translation))
    else:
        ref_point = regularize_transformation(ref_point)
        jacobian = jacobian_translation(ref_point, left_or_right='left')
        transfo_near_id = compose(inverse(ref_point), transfo)
        tangent_vector_from_id = group_log(transfo_near_id)
        tangent_vector = np.dot(jacobian, tangent_vector_from_id)

    return tangent_vector


def inner_product(coef_rotations, coef_translations):
    """
    Compute a 6x6 diagonal matrix, where the diagonal is formed by:
    coef_rotations * [1, 1, 1] and coef_translations * [1, 1, 1].

    :param coef_rotations: scalar
    :param coef_translations: scalar
    :returns inner_product_mat: 6x6 matrix
    """
    inner_product_mat = np.zeros([6, 6])
    inner_product_mat[0:3, 0:3] = coef_rotations * np.eye(3)
    inner_product_mat[3:6, 3:6] = coef_translations * np.eye(3)
    return inner_product_mat


def riemannian_metric(ref_point,
                      inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                      left_or_right='left'):
    """
    Compute the 6x6 matrix of the Riemmanian metric at point ref_point,
    by translating inner_product from the identity to ref_point.

    :param ref_point: 6D vector element of SE(3)
    :param inner_product: 6x6 matrix of inner product at the identity
    :param left_or_right: left/right translation of the inner product
    :returns metric_mat: 6x6 matrix of Riemannian metric at ref_point
    """
    assert left_or_right in ('left', 'right')
    ref_point = regularize_transformation(ref_point)

    jacobian = jacobian_translation(ref_point, left_or_right=left_or_right)
    inv_jacobian = np.linalg.inv(jacobian)

    inv_jacobian_transposed = np.linalg.inv(jacobian.transpose())

    metric_mat = np.dot(inv_jacobian_transposed, inner_product)
    metric_mat = np.dot(metric_mat, inv_jacobian)

    return metric_mat


def riemannian_exp(tangent_vec,
                   inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                   left_or_right='left',
                   ref_point=GROUP_IDENTITY):
    """
    Compute the Riemannian exponential at point ref_point
    of vector tangent vector for left or right translation
    of inner_product.
    This gives a point in SE(3).

    :param tangent_vec: translation part, rotation part of tangent vector
    :param inner_product: matrix of the inner product on the Lie algebra
    :param left_or_right: left or right translation of the inner product
    :param ref_point: 6D vector
    :returns transfo_exp: 6D vector, representing a point

    TODO(nina): Check formulae for right-invariant case.
    """
    def riemannian_exp_from_id(
            tangent_vec,
            inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
            left_or_right='left'):
        if left_or_right == 'left':
            transfo_exp_from_id = np.dot(inner_product, tangent_vec)
            transfo_exp_from_id = regularize_transformation(
                                                transfo_exp_from_id)
        else:
            vec = tangent_vec
            vec[3:6] = -tangent_vec[3:6]
            transfo_exp_from_id = inverse(riemannian_exp_from_id(
                                                vec,
                                                inner_product=inner_product,
                                                left_or_right='left'))
        transfo_exp_from_id = regularize_transformation(transfo_exp_from_id)
        return transfo_exp_from_id

    assert len(ref_point) == 6 & len(tangent_vec) == 6
    assert left_or_right in ('left', 'right')

    ref_point = regularize_transformation(ref_point)
    tangent_vec = regularize_transformation(tangent_vec)

    jacobian = jacobian_translation(ref_point,
                                    left_or_right=left_or_right)
    inv_jacobian = np.linalg.inv(jacobian)

    tangent_vec_translated_to_id = np.dot(inv_jacobian, tangent_vec)
    tangent_vec_translated_to_id = regularize_transformation(
                                       tangent_vec_translated_to_id)

    transfo_exp_from_id = riemannian_exp_from_id(
                                   tangent_vec_translated_to_id,
                                   inner_product=inner_product,
                                   left_or_right=left_or_right)
    if left_or_right == 'left':
        transfo_exp = compose(ref_point, transfo_exp_from_id)

    else:
        transfo_exp = compose(transfo_exp_from_id, ref_point)

    transfo_exp = regularize_transformation(transfo_exp)
    return transfo_exp


def riemannian_log(point,
                   inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                   left_or_right='left',
                   ref_point=GROUP_IDENTITY):
    """
    Compute the Riemannian point at point ref_point
    of point for left or right translation
    of inner_product.
    This gives a point in SE(3).

    :param point: 6D vector, point whose log is taken
    :param inner_product: matrix of the inner product on the Lie algebra
    :param left_or_right: left or right translation of the inner product
    :param ref_point: 6D vector
    :returns transfo_exp: 6D vector, representing a point

    TODO(nina): Check formulae for right-invariant case.
    """
    def riemannian_log_from_id(
            tangent_vec,
            inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
            left_or_right='left'):
        inv_inner_product = np.linalg.inv(inner_product)
        if left_or_right == 'left':
            tangent_vec_log_from_id = np.dot(inv_inner_product, tangent_vec)
        else:
            tangent_vec_log_from_id = np.zeros(6)
            inv_rot_mat = rotations.rotation_matrix_from_rotation_vector(
                    -tangent_vec[0:3])
            tangent_vec_log_from_id[0:3] = -tangent_vec[0:3]
            tangent_vec_log_from_id[3:6] = np.dot(inv_rot_mat,
                                                  tangent_vec[3:6])
            tangent_vec_log_from_id = np.dot(inv_inner_product,
                                             tangent_vec_log_from_id)

        tangent_vec_log_from_id = regularize_transformation(
                                      tangent_vec_log_from_id)
        return tangent_vec_log_from_id

    assert len(ref_point) == 6 & len(point) == 6
    assert left_or_right in ('left', 'right')

    ref_point = regularize_transformation(ref_point)
    point = regularize_transformation(point)

    if left_or_right == 'left':
        point_near_id = compose(inverse(ref_point), point)

    else:
        point_near_id = compose(point, inverse(ref_point))

    transfo_vec_log_from_id = riemannian_log_from_id(
                                   point_near_id,
                                   inner_product=inner_product,
                                   left_or_right=left_or_right)
    jacobian = jacobian_translation(ref_point,
                                    left_or_right=left_or_right)
    transfo_vec_log = np.dot(jacobian, transfo_vec_log_from_id)
    transfo_vec_log = regularize_transformation(transfo_vec_log)
    return transfo_vec_log


def square_riemannian_norm(tangent_vector,
                           left_or_right='left_or_right',
                           inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                           ref_point=GROUP_IDENTITY):
    """
    Compute the square norm of a tangent vector,
    using the riemannian metric at ref_point
    defined by the left or right translation of
    inner_product at the identity.

    :param tangent_vector: 6D vector whose square norm is computed
    :param left_or_right:
    :param inner_product:
    :param ref_point:

    :returns sq_riem_norm: positive scalar, squared norm of tangent_vector

    """
    assert len(tangent_vector) == 6 & len(ref_point) == 6
    assert left_or_right in ('left', 'right')

    tangent_vector = regularize_transformation(tangent_vector)
    ref_point = regularize_transformation(ref_point)

    riem_metric_mat = riemannian_metric(ref_point=ref_point,
                                        inner_product=inner_product,
                                        left_or_right=left_or_right)
    sq_riem_norm = np.dot(np.dot(tangent_vector.transpose(),
                                 riem_metric_mat),
                          tangent_vector)

    return sq_riem_norm


def random_uniform():
    """
    Generate an 6d vector element of SE(3) uniformly

    :returns random transfo: 6d vector element of SE(3)
    """
    random_rot_vec = np.random.rand(3) * 2 - 1
    random_rot_vec = rotations.regularize_rotation_vector(random_rot_vec)
    random_translation = np.random.rand(3) * 2 - 1

    random_transfo = np.concatenate([random_rot_vec, random_translation])
    return random_transfo


def exponential_matrix(rot_vec, epsilon=1e-5):
    """
    Compute the exponential of the rotation matrix
    represented by rot_vec.

    :param rot_vec: 3D rotation vector
    :returns exponential_mat: 3x3 matrix
    """

    rot_vec = rotations.regularize_rotation_vector(rot_vec)

    angle = np.linalg.norm(rot_vec)
    skew_rot_vec = rotations.skew_matrix_from_vector(rot_vec)

    if angle == 0:
        coef_1 = 0
        coef_2 = 0

    elif angle < epsilon:
        coef_1 = 1. / 2. - angle ** 2 / 24.
        coef_2 = 1. / 6. - angle ** 3 / 120.

    else:
        coef_1 = angle ** (-2) * (1. - np.cos(angle))
        coef_2 = angle ** (-2) * (1. - np.sin(angle) / angle)

    exponential_mat = (np.eye(3)
                       + coef_1 * skew_rot_vec
                       + coef_2 * np.dot(skew_rot_vec, skew_rot_vec))

    return exponential_mat


def exponential_barycenter(transfo_vectors, weights):
    """

    :param transfo_vectors: SE3 data points, Nx6 array
    :param weights: data point weights, Nx1 array
    """

    n_transformations, _ = transfo_vectors.shape

    if n_transformations < 2:
        raise ValueError('Requires # of transformations >=2.')

    rotation_vectors = transfo_vectors[:, 0:3]
    translations = transfo_vectors[:, 3:6]

    mean_rotation = rotations.riemannian_mean(rotation_vectors, weights)

    # Partie translation, p34 de expbar
    matrix = np.zeros([3, 3])
    translation = np.zeros(3)

    for i in range(0, n_transformations):
        rot_vec_i = rotation_vectors[i, :]
        translation_i = translations[i, :]
        weight_i = weights[i]

        inv_rot_mat_i = rotations.rotation_matrix_from_rotation_vector(
                -rot_vec_i)
        mean_rotation_mat = rotations.rotation_matrix_from_rotation_vector(
                mean_rotation)

        matrix_aux = np.dot(mean_rotation_mat, inv_rot_mat_i)
        vec_aux = rotations.rotation_vector_from_rotation_matrix(matrix_aux)
        matrix_aux = exponential_matrix(vec_aux)
        matrix_aux = np.linalg.inv(matrix_aux)

        matrix = matrix + weight_i * matrix_aux
        translation = (translation
                       + weight_i * np.dot(np.dot(matrix_aux,
                                                  inv_rot_mat_i),
                                           translation_i))

    mean_transformation = np.zeros(6)
    mean_transformation[0:3] = mean_rotation
    mean_transformation[3:6] = np.dot(np.linalg.inv(matrix), translation)
    return mean_transformation


def riemannian_variance(ref_point, transfo_vectors, weights,
                        left_or_right='left',
                        inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT):
    """
    Computes the variance of the transformations in transfo_vectors
    at the point ref_point.

    The variance is computed using weighted squared geodesic
    distances from ref_rotation_vector to the data.
    The geodesic distance is the left- (resp. right-) invariant
    Riemannian distance.

    :param ref_point: point at which to compute the variance
    :param transfo_vectors: array of rotation vectors
    :param weights: array of corresponding weights
    :returns variance: variance of transformations at ref_point

    """
    assert left_or_right in ('left', 'right')

    n_transformations, _ = transfo_vectors.shape

    if n_transformations < 2:
        logging.info('Error: Calculating variance requires at least 2 points')
        return 0

    variance = 0

    for i in range(1, n_transformations):
        transfo_i = transfo_vectors[i, :]
        weight_i = weights[i]

        log_vec = riemannian_log(transfo_i,
                                 left_or_right=left_or_right,
                                 ref_point=ref_point,
                                 inner_product=inner_product)
        sq_riem_norm = square_riemannian_norm(log_vec,
                                              left_or_right=left_or_right,
                                              inner_product=inner_product,
                                              ref_point=ref_point)
        variance += weight_i * sq_riem_norm

    return variance


def riemannian_mean(transfo_vectors, weights,
                    left_or_right='left',
                    inner_product=ALGEBRA_CANONICAL_INNER_PRODUCT,
                    epsilon=1e-5):
    """
    Computes the weighted mean of the
    transformations in transfo_vectors

    The geodesic distances are obtained by the
    left-invariant Riemannian distance.

    :param transfo_vectors: array of 3d transformations
    :param weights: array of weights
    :returns riem_mean: 3d vector, weighted mean of transfo_vectors
    """
    assert left_or_right in ('left', 'right')

    n_transformations, _ = transfo_vectors.shape

    if n_transformations < 2:
        logging.info('Error: Calculating mean requires at least 2 points.')

    riem_mean = transfo_vectors[0, :]

    while True:
        aux = np.zeros(6)
        riem_mean_next = riem_mean
        for i in range(0, n_transformations):
            transfo_i = transfo_vectors[i, :]
            weight_i = weights[i]

            riem_log_i = riemannian_log(transfo_i,
                                        left_or_right=left_or_right,
                                        inner_product=inner_product,
                                        ref_point=riem_mean_next)
            aux += weight_i * riem_log_i
        riem_mean = riemannian_exp(aux,
                                   ref_point=riem_mean_next,
                                   left_or_right=left_or_right,
                                   inner_product=inner_product)

        riem_mean_vector_diff = riemannian_log(riem_mean,
                                               ref_point=riem_mean_next,
                                               left_or_right=left_or_right,
                                               inner_product=inner_product)
        riem_mean_diff = square_riemannian_norm(riem_mean_vector_diff)
        variance = riemannian_variance(riem_mean_next,
                                       transfo_vectors,
                                       weights,
                                       inner_product=inner_product)

        if riem_mean_diff < epsilon * variance:
            break

    return riem_mean
