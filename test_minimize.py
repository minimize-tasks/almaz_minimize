from unittest import TestCase

from scipy.optimize import minimize
import numpy as np

import radar_potential_saushkin as rpt_saushkin


def wrap_(func, minimize_args, func_result, init_values_dict=None, args_bounds=None):
    """"""
    funccc = wrapper_for_minimization(func, minimize_args, func_result, init_values_dict)

    if args_bounds is None:
        args_bounds = len(minimize_args) * ((1e-8, 1e6),)

    for _ in range(1000):
        result = minimize(funccc, 1e0*np.random.rand(len(minimize_args)), tol=1e-1,
                          bounds=args_bounds)
        if result.success and np.allclose(result.fun + func_result, func_result, rtol=0.001):
            print(_, result.fun)
            return result.x


def wrapper_for_minimization(func, minimize_args, func_result, init_values_dict=None):
    """"""
    if init_values_dict is None:
        init_values_dict = dict()

    funccc = lambda args: np.power(
        func_result - func(
            **dict(zip(minimize_args, args)),
            **init_values_dict),
        2)
    return funccc


class MyTest(TestCase):
    def test_wrap_(self):
        func = rpt_saushkin.real_potential_rls

        init_values_dict = dict(pt=10000, gain_db=30, wavelength=7, loss_db=1, F=1,
                                p_min=0.001)
        func_result = func(**init_values_dict)
        init_values_dict2 = dict(pt=10000, gain_db=26.35181398, wavelength=16.21498837,
                                 loss_db=1, F=1, p_min=0.001)
        func_result2 = func(**init_values_dict2)
        print(func_result, func_result2)
        new_fixed_values = dict(loss_db=1, F=1, p_min=0.001, pt=10000)
        minimize_args = ("wavelength", "gain_db")


        wrap_result = wrap_(func, minimize_args, func_result=func_result,
                            init_values_dict=new_fixed_values)
        if wrap_result is not None:
            minimize_args_dict = dict(zip(minimize_args, wrap_result))

            func_result3 = func(**new_fixed_values, **minimize_args_dict)
            print(func_result, 'optimization_res= ', func_result3)
        print(wrap_result)
