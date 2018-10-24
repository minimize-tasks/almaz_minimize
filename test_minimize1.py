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
    func_ = wrapper_for_minimization(funcs, variable_parameters,
                                     y_values, fixed_x_dict)

    if bounds is None:
        bounds = len(variable_parameters[0]) * ((0, None),)
    cons = []
    for i in range(len(bounds[0])):
        cons = cons + [lambda x, i_local=i: x[i_local] - bounds[0][i_local][0]]
        cons = cons + [lambda x, i_local=i: -x[i_local] + bounds[0][i_local][1]]
    for _ in range(1000):
        result = fmin_cobyla(func_, 1e0 * np.random.rand(len(variable_parameters[0])), cons=cons)#, rhoend=1)
        return dict(zip(variable_parameters[0], result))

def func1(args,funcs, variable_parameters, y_values, fixed_x_dict):
    arr = np.zeros(len(funcs))
    for i in range(len(funcs)):
        arr[i] = np.power(y_values[i] - funcs[i](**dict(zip(variable_parameters[i], args)),**fixed_x_dict[i]),2)
    return arr


def wrapper_for_minimization(func, variable_parameters, y_values, fixed_x_dict):
    """"""

    funccc = lambda args: np.sum(func1(args,func, variable_parameters, y_values, fixed_x_dict))
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
        new_fixed_values1 = dict(loss_db=1.2, F=1.15, p_min=0.001, pt=10000)
        minimize_args = ("wavelength", "gain_db")

        wrap_result = wrap_([func,func, ], [minimize_args, minimize_args, ], y_values=[func_result, 89, ],
                            fixed_x_dict=[new_fixed_values,new_fixed_values1, ],
                            bounds=[[(0, 4), (20, 30)], ])
        if wrap_result is not None:

            func_result3 = func(**new_fixed_values, **wrap_result)
            print("\n")
            print(func_result, 'res1= ', func_result3)
            func_result4 = func(**new_fixed_values1, **wrap_result)
            print('89  res2= ', func_result4)
        print(wrap_result)
