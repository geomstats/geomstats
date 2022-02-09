import inspect
import types

import properties
import pytest


def _iterate_and_assign(attrs, properties_class):
    for attr_name, attr_value in properties_class.__dict__.items():
        if isinstance(attr_value, types.FunctionType):
            attrs["test_" + attr_name] = attr_value

    return attrs


class Parametrizer(type):
    """Metaclass for test files.

    Parametrizer decorates every function inside the class with pytest.mark.parametrizer
    (except class methods and static methods). Two conventions need to be respected:

    1.There should be a TestData object named 'testing_data'.
    2.Every test function should have its corresponding data function inside
    TestData object.

    Ex. test_exp() should have method called exp_data() inside 'testing_data'.
    """

    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):

                args_str = ", ".join(inspect.getfullargspec(attr_value)[0][1:])
                data_fn_str = attr_name[5:] + "_data"
                attrs[attr_name] = pytest.mark.parametrize(
                    args_str,
                    getattr(locals()["attrs"]["testing_data"], data_fn_str)(),
                )(attr_value)

        return super(Parametrizer, cls).__new__(cls, name, bases, attrs)


class ManifoldParametrizer(Parametrizer):
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.ManifoldProperties)
        return super(ManifoldParametrizer, cls).__new__(cls, name, bases, attrs)


class OpenSetParametrizer(ManifoldParametrizer):
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.OpenSetProperties)
        return super(OpenSetParametrizer, cls).__new__(cls, name, bases, attrs)


class LieGroupParametrizer(ManifoldParametrizer):
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.LieGroupProperties)
        return super(LieGroupParametrizer, cls).__new__(cls, name, bases, attrs)


class VectorSpaceParametrizer(ManifoldParametrizer):
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.VectorSpaceProperties)
        return super(VectorSpaceParametrizer, cls).__new__(cls, name, bases, attrs)


class MatrixLieAlgebraParametrizer(VectorSpaceParametrizer):
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.MatrixLieAlgebraProperties)
        return super(MatrixLieAlgebraParametrizer, cls).__new__(cls, name, bases, attrs)


class LevelSetParametrizer:
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.LevelSetProperties)
        return super(LevelSetParametrizer, cls).__new__(cls, name, bases, attrs)


class ConnectionParametrizer:
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.ConnectionProperties)
        return super(ConnectionParametrizer, cls).__new__(cls, name, bases, attrs)


class RiemannianMetricParametrizer:
    def __new__(cls, name, bases, attrs):
        _iterate_and_assign(attrs, properties.RiemannianMetricProperties)
        return super(RiemannianMetricParametrizer, cls).__new__(cls, name, bases, attrs)