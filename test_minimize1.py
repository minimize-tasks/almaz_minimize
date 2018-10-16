from unittest import TestCase

from scipy.optimize import fmin_cobyla
import numpy as np

import radar_potential_saushkin as rpt_saushkin


def wrap_(funcs, variable_parameters, y_values, fixed_x_dict, bounds=None):
    """

    Args:
        funcs (list(func, )): List with functions.
        variable_parameters  (list(list, )): List with names varible parameters.
        y_values  (list(floats, )): List with 'y' values.
        fixed_x_dict  (list(dict, )): Dict with init parameters ({'x':1, 'xx':2})
        bounds  (list(tuple, )|None):  Bounds.

    Returns (list(dict, )|None):

    """
    func_ = wrapper_for_minimization(funcs[0], variable_parameters[0],
                                     y_values[0], fixed_x_dict[0])

    if bounds is None:
        bounds = len(variable_parameters[0]) * ((0, None),)
    cons = []
    for i in range(len(bounds[0])):
        cons = cons + [lambda x, i_local=i: x[i_local] - bounds[0][i_local][0]]
        cons = cons + [lambda x, i_local=i: -x[i_local] + bounds[0][i_local][1]]
    for _ in range(1000):
        result = fmin_cobyla(func_, 1e0 * np.random.rand(len(variable_parameters[0])), cons=cons)
        return dict(zip(variable_parameters[0], result))

def wrapper_for_minimization(func, variable_parameters, y_values, fixed_x_dict):
    """"""

    funccc = lambda args: np.power(
        y_values - func(
            **dict(zip(variable_parameters, args)),
            **fixed_x_dict),
        2)
    return funccc


class MyTest(TestCase):
    def test_wrap_(self):
        func = rpt_saushkin.real_potential_rls

        init_values_dict = dict(pt=10000, gain_db=25, wavelength=1, loss_db=1, F=1,
                                p_min=0.001)
        func_result = func(**init_values_dict)
        init_values_dict2 = dict(pt=10000, gain_db=26.35181398, wavelength=16.21498837,
                                 loss_db=1, F=1, p_min=0.001)
        func_result2 = func(**init_values_dict2)
        new_fixed_values = dict(loss_db=1, F=1, p_min=0.001, pt=10000)
        minimize_args = ("wavelength", "gain_db")

        wrap_result = wrap_([func, ], [minimize_args, ], y_values=[func_result, ],
                            fixed_x_dict=[new_fixed_values, ], bounds=[[(0, 4), (20, 30)], ])
        if wrap_result is not None:

            func_result3 = func(**new_fixed_values, **wrap_result)
            print("\n")
            print(func_result, 'optimization_res= ', func_result3)
        print(wrap_result)
