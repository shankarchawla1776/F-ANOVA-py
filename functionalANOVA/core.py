from .one_way import one_way
from .one_way_BF import one_way_BF
from .two_way import two_way
from .two_way_BF import two_way_BF
from .two_group_cov import two_group_cov
from .k_group_cov import k_group_cov
from .k_group_cov_pairwise import k_group_cov_pairwise
from .function_subsetter import function_subsetter

class FunctionalANOVA:
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

    def one_way_anova(self, **kwargs):
        return one_way(self, **kwargs)

    def one_way_anova_bf(self, **kwargs):
        return one_way_BF(self, **kwargs)

    def two_way_anova(self, **kwargs):
        return two_way(self, **kwargs)

    def two_way_anova_bf(self, **kwargs):
        return two_way_BF(self, **kwargs)

    def two_group_covariance(self, **kwargs):
        return two_group_cov(self, **kwargs)

    def k_group_covariance(self, **kwargs):
        return k_group_cov(self, **kwargs)

    def k_group_covariance_pairwise(self, **kwargs):
        return k_group_cov_pairwise(self, **kwargs)

    def function_subsetter(self):
        return function_subsetter(self)

    # accept downstream attribute errors for missing arguments
    # def one_way_anova(self, *args, **kwargs):
    #     try:
    #         return one_way(self, *args, **kwargs)
    #     except AttributeError as e:
    #         missing_var = e
    #         raise ValueError(
    #             f"'{missing_var}' must be defined for the 'one_way_anova' method"
    #         )
