from unittest import TestCase
import unittest
from scipy.optimize import fmin_cobyla
import numpy as np

import radar_potential_saushkin as rpt_saushkin
from test_minimize1 import wrap_


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

    def test_call_without_args(self):
        """ Проверка, что без при вызове без аргументов вознкает TypeError, если ошибки
            не было, тест завершится с assert

        Returns:

        """
        try:
            wrap_()
        except TypeError:
            pass
        else:
            assert False

    def test_call_with_empty_list(self):
        """ Проверка, можно ли вызывать функцию с пустыми списами. В докстринге wrap_
            нет на это ограничений.

            Видим что падает с IndexError. Значит нужно добавить проверок в wrap_,
            или хотябы поправить описание.
        """
        wrap_([], [], [], [], [])

    def test_wrap_3(self):
        func1 = rpt_saushkin.real_potential_rls
        
        init_values_dict1 = dict(pt=20000, gain_db=20, wavelength=0.00002, loss_db=2, p_min=15000, F=1)
        func1_result = func1(**init_values_dict1)
        
        func2 = rpt_saushkin.std_h
        
        init_values_dict2 = dict(distance=1000, elevation=2, elevation_rls=1.5, snr_db=5, tau=0.002,
                                 aperture=7, wavelength=0.00002, width_spec=None, tau_disc_fcm=None, mode='nonmod')
        func2_result = func2(**init_values_dict2)

        new_fixed_values1 = dict(pt=20000, gain_db=20, loss_db=2, p_min=15000, F=1)
        new_fixed_values2 = dict(distance=1000, elevation=2, elevation_rls=1.5, snr_db=5, tau=0.002, aperture=7,
                             width_spec=None, tau_disc_fcm=None, mode='nonmod')
        
        minimize_args = (['wavelength'])

        wrap_result = wrap_([func1, func2], [minimize_args, minimize_args], y_values=[func1_result, func2_result],
                            fixed_x_dict=[new_fixed_values1, new_fixed_values2],
                            bounds=[[(0.00001, 0.00006)], [(0.00003, 0.00008)]])
        if wrap_result is not None:
            func_result3 = func1(**new_fixed_values1, **wrap_result)
            print("\n")
            print(func1_result, 'res1= ', func_result3)
            func_result4 = func2(**new_fixed_values2, **wrap_result)
            print(func2_result, '  res2= ', func_result4)
        print(wrap_result)


