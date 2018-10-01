# -*- coding: utf-8 -*-
"""Tests phased.radar_potential."""
import unittest
import numpy as np

import radar_potential_saushkin as rpt_suashkin


class PotentialSauskinTest(unittest.TestCase):

    def test_rotation_period(self):
        period = rpt_suashkin.rotation_period(2, 1.9, 2, 1, 10)
        self.assertEqual(period, 40)

    def test_radareqrange(self):
        limit = rpt_suashkin.fixed_probability_limit(0.9, 0.001, 1)
        range = rpt_suashkin.radareqrange(209, 1, limit)
        np.testing.assert_allclose(range, 99713.0514717)

    def test_fixed_probability_limit(self):
        limit = rpt_suashkin.fixed_probability_limit(0.9, 0.001, 1)
        np.testing.assert_allclose(limit, 8.035113)

    def test_potential_rls(self):
        potential_db = rpt_suashkin.potential_rls(1e5, 1, 8)
        self.assertEqual(potential_db, 209.03089986991944,
                         "Potential in dB = {}".format(potential_db))

    def test_real_potential_rls(self):
        potential_db = rpt_suashkin.real_potential_rls(1e5, 20, 0.3, 0, 1, 5e-16)
        # TODO. Random equals
        np.testing.assert_allclose(potential_db, 199.5764291303)

    def test_gain(self):
        gain = rpt_suashkin.gain(10, 0.03)
        np.testing.assert_allclose(gain, 51.44967)

    def test_norm_directivity(self):
        n_d1 = rpt_suashkin.norm_directivity(np.deg2rad(0), np.deg2rad(1))
        np.testing.assert_allclose(n_d1, 1)
        n_d2 = rpt_suashkin.norm_directivity(np.deg2rad(10), np.deg2rad(10))
        np.testing.assert_allclose(n_d2, 0.04321391826377226)

    def test_beam_width_by_aperture(self):
        b_w = rpt_suashkin.beam_width_by_aperture(0.3, 3)
        self.assertEqual(b_w, 0.88 * 0.3 / 3,
                         'beam in degree = {}'.format(np.rad2deg(b_w)))

    def test_probability_detect_by_antenna_param(self):
        pfa = 0.001
        pd = 0.9
        potential_db = 209
        rcs = 1
        num_pulse = 3
        k = 0.6
        limit = rpt_suashkin.fixed_probability_limit(pd, pfa, num_pulse, k)
        range = rpt_suashkin.radareqrange(potential_db, rcs, limit)

        new_pd = rpt_suashkin.probability_detect_by_antenna_param(potential_db, range,
                                                                  rcs, pfa, num_pulse, k)
        self.assertEqual(pd, new_pd)

    def test_probability_detect_by_range(self):
        distance = np.arange(0, 400, 50)
        distance_max = 200
        pd = rpt_suashkin.probability_detect_by_range(distance, distance_max)
        self.assertEqual(pd[0], 1)
        self.assertEqual(pd[-1], 8.448756028504651e-05)
        pd = rpt_suashkin.probability_detect_by_range(distance[-1], distance_max)
        self.assertEqual(pd, 8.448756028504651e-05)

    def test_sko_angle(self):
        sko = rpt_suashkin.std_angle(0.3, 3, 0, 13)
        np.testing.assert_allclose(sko, 0.013432326831410)

    def test_sko_distance(self):
        sko = rpt_suashkin.std_distance(13, tau=1e-5, mode='nonmod')
        self.assertIsNotNone(sko)
        sko = rpt_suashkin.std_distance(13, width_spec=1e5, mode='lfm')
        self.assertIsNotNone(sko)
        sko = rpt_suashkin.std_distance(13, tau_disc_fcm=1e-4, mode='fcm')
        self.assertIsNotNone(sko)

    def test_sko_speed(self):
        sko = rpt_suashkin.std_speed(13, wavelength=0.3, tau=1e-5, mode='nonmod')
        self.assertIsNotNone(sko)
        sko = rpt_suashkin.std_speed(13, wavelength=0.3, width_spec=1e5, T_lfm=1e-4,
                                     mode='lfm')
        self.assertIsNotNone(sko)
        sko = rpt_suashkin.std_speed(13, wavelength=0.3, tau=1e-5, mode='fcm')
        self.assertIsNotNone(sko)

    def test_sko_x(self):
        distance = 1e5
        elevation = 1
        elevation_rls = .2
        azimuth = 2
        azimuth_rls = 0.1
        std_dist = 100
        std_el = std_az = 0.1
        sko_x_ = rpt_suashkin.std_x(distance, elevation, elevation_rls, azimuth,
                                    azimuth_rls, std_dist, std_el, std_az)
        sko_y_ = rpt_suashkin.std_y(distance, elevation, elevation_rls, azimuth,
                                    azimuth_rls, std_dist, std_el, std_az)
        sko_xy_ = rpt_suashkin.std_xy(distance, elevation)
        self.assertIsNotNone(sko_x_)
        self.assertIsNotNone(sko_y_)
        self.assertIsNotNone(sko_xy_)
