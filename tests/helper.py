"""
Helper functions for unit tests.
"""

import geomstats.backend as gs


def to_scalar(expected):
    expected = gs.to_ndarray(expected, to_ndim=1)
    expected = gs.to_ndarray(expected, to_ndim=2, axis=-1)
    return expected


def to_vector(expected):
    expected = gs.to_ndarray(expected, to_ndim=2)
    return expected


def to_matrix(expected):
    expected = gs.to_ndarray(expected, to_ndim=3)
    return expected


def left_log_then_exp_from_identity(metric, point):
    aux = metric.left_log_from_identity(point=point)
    result = metric.left_exp_from_identity(tangent_vec=aux)
    return result


def left_exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.left_exp_from_identity(tangent_vec=tangent_vec)
    result = metric.left_log_from_identity(point=aux)
    return result


def log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.exp_from_identity(tangent_vec=tangent_vec)
    result = metric.log_from_identity(point=aux)
    return result


def log_then_exp(metric, point, base_point=None):
    aux = metric.log(point=point,
                     base_point=base_point)
    result = metric.exp(tangent_vec=aux,
                        base_point=base_point)
    return result


def exp_then_log(metric, tangent_vec, base_point=None):
    aux = metric.exp(tangent_vec=tangent_vec,
                     base_point=base_point)
    result = metric.log(point=aux,
                        base_point=base_point)
    return result


def group_log_then_exp_from_identity(group, point):
    aux = group.log_from_identity(point=point)
    result = group.exp_from_identity(tangent_vec=aux)
    return result


def group_exp_then_log_from_identity(group, tangent_vec):
    aux = group.exp_from_identity(tangent_vec=tangent_vec)
    result = group.log_from_identity(point=aux)
    return result


def group_log_then_exp(group, point, base_point):
    aux = group.log(
        point=point, base_point=base_point)
    result = group.exp(
        tangent_vec=aux, base_point=base_point)
    return result


def group_exp_then_log(group, tangent_vec, base_point):
    aux = group.exp(
        tangent_vec=tangent_vec, base_point=base_point)
    result = group.log(
        point=aux, base_point=base_point)
    return result


def test_parallel_transport(space, metric, shape):
    results = []

    def is_isometry(tan_a, trans_a, endpoint):
        is_tangent = space.is_tangent(trans_a, endpoint)
        is_equinormal = gs.isclose(
            metric.norm(trans_a, endpoint), metric.norm(tan_a, endpoint))
        return gs.logical_and(is_tangent, is_equinormal)

    base_point = space.random_point(shape[0])
    tan_vec_a = space.to_tangent(gs.random.rand(*shape) / 5, base_point)
    tan_vec_b = space.to_tangent(gs.random.rand(*shape) / 5, base_point)
    end_point = metric.exp(tan_vec_b, base_point)

    transported = metric.parallel_transport(
        tan_vec_a, tan_vec_b, base_point)
    result = is_isometry(tan_vec_a, transported, end_point)
    results.append(gs.all(result))

    base_point = base_point[0]
    tan_vec_a = space.to_tangent(tan_vec_a, base_point)
    tan_vec_b = space.to_tangent(tan_vec_b, base_point)
    end_point = metric.exp(tan_vec_b, base_point)
    transported = metric.parallel_transport(
        tan_vec_a, tan_vec_b, base_point)
    result = is_isometry(tan_vec_a, transported, end_point)
    results.append(gs.all(result))

    one_tan_vec_a = tan_vec_a[0]
    transported = metric.parallel_transport(
        one_tan_vec_a, tan_vec_b, base_point)
    result = is_isometry(one_tan_vec_a, transported, end_point)
    results.append(gs.all(result))

    one_tan_vec_b = tan_vec_b[0]
    end_point = end_point[0]
    transported = metric.parallel_transport(
        tan_vec_a, one_tan_vec_b, base_point)
    result = is_isometry(tan_vec_a, transported, end_point)
    results.append(gs.all(result))

    transported = metric.parallel_transport(
        one_tan_vec_a, one_tan_vec_b, base_point)
    result = is_isometry(one_tan_vec_a, transported, end_point)
    results.append(gs.all(result))

    transported = metric.parallel_transport(
        one_tan_vec_a, gs.zeros_like(one_tan_vec_b), base_point)
    result = gs.isclose(transported, one_tan_vec_a)
    results.append(gs.all(result))

    return results


def test_projection_and_belongs(space, shape, atol=gs.atol):
    result = []

    point = gs.random.normal(size=shape)
    projected = space.projection(point)
    belongs = space.belongs(projected, atol=atol)
    result.append(gs.all(belongs))

    point = point[0]
    projected = space.projection(point)
    belongs = space.belongs(projected, atol=atol)
    result.append(belongs)

    point = space.random_point()
    projected = space.projection(point)
    result.append(gs.allclose(point, projected, atol=atol))
    return result


def test_to_tangent_is_tangent(space, atol=gs.atol):
    result = []

    point = space.random_point(2)
    vector = gs.random.rand(*point.shape)
    tangent = space.to_tangent(vector, point)
    is_tangent = space.is_tangent(tangent, point, atol)
    result.append(gs.all(is_tangent))

    vector = gs.random.rand(*point.shape)
    tangent = space.to_tangent(vector[0], point[0])
    is_tangent = space.is_tangent(tangent, point[0], atol)
    result.append(is_tangent)

    projection = space.to_tangent(tangent, point[0])
    result.append(gs.allclose(projection, tangent, atol))
    return result
