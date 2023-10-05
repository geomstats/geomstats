from .multinomial import MultinomialMetricTestData


class CategoricalMetricTestData(MultinomialMetricTestData):
    xfails = ("log_after_exp",)
