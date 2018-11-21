from unittest import TestCase
import unittest
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
    spisok=[]
    spisok.append(funcs)
    spisok.append(variable_parameters)
    spisok.append(y_values)
    spisok.append(fixed_x_dict)

    list_var = list()
    boundsss = list()
    for i in range(len(variable_parameters)):
        for j in range(len(variable_parameters[i])):
            if(list_var.count(variable_parameters[i][j]) == 0):
                list_var.append(variable_parameters[i][j])
                boundsss.append(list(bounds[i][j]))
            for k in range(len(list_var)):
                if(list_var[k] == variable_parameters[i][j]):
                    if(boundsss[k][0] < bounds[i][j][0]):
                        boundsss[k][0] = bounds[i][j][0]
                    if(boundsss[k][1] > bounds[i][j][1]):
                        boundsss[k][1] = bounds[i][j][1]

    func_ = wrapper_for_minimization(funcs, variable_parameters,
                                     y_values, fixed_x_dict, list_var)

    cons = []
    for i in range(len(boundsss)):
        cons = cons + [lambda x, i_local=i: x[i_local] - boundsss[i_local][0]]
        cons = cons + [lambda x, i_local=i: -x[i_local] + boundsss[i_local][1]]
    for _ in range(1000):
        result = fmin_cobyla(func_, 1e0 * np.random.rand(len(variable_parameters[0])), cons=cons)#, rhoend=1)
        return dict(zip(variable_parameters[0], result))


def func1(args,funcs, variable_parameters, y_values, fixed_x_dict, list_var):
    arr = np.zeros(len(funcs))
    for i in range(len(funcs)):
        arg = dict()
        for j in range(len(list_var)):
            for k in range(len(variable_parameters[i])):
                if(list_var[j] == variable_parameters[i][k]):
                    arg.update({list_var[j] : args[j]})
        arr[i] = np.power(y_values[i] - funcs[i](**arg,**fixed_x_dict[i]),2)
    return arr


def wrapper_for_minimization(func, variable_parameters, y_values, fixed_x_dict, list_var):
    funccc = lambda args: np.sum(func1(args,func, variable_parameters, y_values, fixed_x_dict, list_var))
    return funccc


class MyTest(TestCase):
    '''
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

        wrap_result = wrap_([func,func], [minimize_args, minimize_args], y_values=[func_result, 89],
                            fixed_x_dict=[new_fixed_values,new_fixed_values1],
                            bounds=[[(0, 4), (20, 30)], [(0.5, 4.5), (19, 29)]])
        if wrap_result is not None:

            func_result3 = func(**new_fixed_values, **wrap_result)
            print("\n")
            print(func_result, 'res1= ', func_result3)
            func_result4 = func(**new_fixed_values1, **wrap_result)
            print('89  res2= ', func_result4)
        print(wrap_result)
    '''
    def test_wrap_1(self):
        func = rpt_saushkin.rotation_period

        init_values_dict = dict(azimuth_range=80, azimuth_step=2, elevate_range=60, elevation_step=3,
                    time_one_pulse=0.0002, num_ping=1)
        func_result = func(**init_values_dict)
        init_values_dict2 = dict(azimuth_range=60, azimuth_step=1, elevate_range=50, elevation_step=5,
                    time_one_pulse=0.0002, num_ping=1)
        func_result2 = func(**init_values_dict2)
        new_fixed_values = dict(azimuth_range=80, azimuth_step=2, elevation_step=3, num_ping=1)
        new_fixed_values1 = dict(azimuth_range=120, azimuth_step=1, elevation_step=2, num_ping=1)
        minimize_args = ("time_one_pulse", "elevate_range")




        wrap_result = wrap_([func,func], [minimize_args, minimize_args], y_values=[func_result,func_result2],
                            fixed_x_dict=[new_fixed_values,new_fixed_values1],
                            bounds=[[(0.0001, 0.0004), (40, 50)], [(0.0003, 0.0008), (30, 200)]])

        if wrap_result is not None:

            func_result3 = func(**new_fixed_values, **wrap_result)
            print("\n")
            print(func_result, 'res1= ', func_result3)
            func_result4 = func(**new_fixed_values1, **wrap_result)
            print(func_result2,  'res2= ', func_result4)
        print(wrap_result)





