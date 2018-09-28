"""Module for compute radar parameters by saushkin."""

import numpy as np
from numpy import float64
from scipy import special as scs
import scipy.constants as scc

from phased.misc import pow2db, db2pow, systemp, db2mag
from phased.misc import FluctuationModel
from phased.misc import PulseIntegration, MagnitudeUnit, mag2db
from phased.misc import marcumq

EARTH_RADIUS = 6378000


def rotation_period(azimuth_range, azimuth_step, elevate_range, elevation_step,
                    time_one_pulse, num_ping=1):
    """(1) Compute rotation period.

    Args:
        azimuth_range:
        azimuth_step:
        elevate_range:
        elevation_step:
        time_one_pulse:
        num_ping: Ping count in pack

    Returns:

    """
    num_azimuth_step = np.ceil(azimuth_range / azimuth_step)
    num_elevation_step = np.ceil(elevate_range / elevation_step)
    period = num_azimuth_step * num_elevation_step * time_one_pulse * num_ping
    return period


def radareqrange(potential_db, rcs, ro):
    """(3)Max distance.

    Args:
        potential_db (float): In dB
        rcs (float): In m^2
        ro (float): probability limit log(Pa)/log(Pd) ?????????? in dB??????

    Returns:

    """
    potential_rls_power = db2pow(potential_db)
    return np.power(potential_rls_power * rcs / ro, 0.25)


def min_range(tau):
    """(*)Min distance.

    Args:
        tau (float): Pulse length in seconds.

    Returns (float): Min range.

    """
    min_range = 0.5 * scc.c * tau
    return min_range


def fixed_probability_limit(pd, pfa, num_pulse=1, k=1):
    """(4)

    Args:
        pd: Probapility detect
        pfa: Probsbility falsealarm
        num_pulse: Pulse count in pack ???
        k (float): coef. Default 1
    Returns:

    """
    return np.sqrt((np.log(pfa) / np.log(pd) - 1) / np.power(num_pulse, k))


def potential_rls(distance, rcs, ro):
    """(5)

    Args:
        distance: in m
        rcs: in m^2
        ro: probability limit log(Pa)/log(Pd) ?????????? in dB??????

    Returns (float):
        Potential in dB
    """
    return pow2db(np.power(distance, 4) * ro / rcs)


def real_potential_rls(pt, gain_db, wavelength, loss_db, F, p_min):
    """(6)

    Args:
        pt (float): Power in Watts
        gain_db (float): Faint in dB
        wavelength (float): in m
        loss_db (float): Loaa in dB
        F (float): norm DNA. def=1
        p_min: min power in Watts.

    Returns:
        Real potential in dB.
    """
    gain_pow = db2pow(gain_db)
    loss_pow = db2pow(loss_db)
    potential_tmp = pt * gain_pow * gain_pow * np.power(wavelength, 2) * loss_pow * F * F
    potential = potential_tmp / (np.power(4 * np.pi, 3) * p_min)
    return pow2db(potential)


def gain(S, wavelength):
    """(8)

    Args:
        S: area in m^2
        wavelength: in m

    Returns:
        return gain in dB.
    """
    return pow2db(4 * np.pi * S / np.power(wavelength, 2))


def effective_area(hight, width, k):
    """(11)

    Args:
        hight: in m
        width: in m
        k:

    Returns:

    """
    return hight * width * k


def norm_directivity(alpha, beamwidths):
    """(17)

    Args:
        alpha: Target angle in anten system. (Radian)
        beamwidths: with 0.5 power (Radian)

    Returns:

    """
    return np.exp(-np.pi * np.power(alpha / beamwidths, 2))


def beam_width_by_aperture(wavelength, aperture, alpha=0, K=0.88):
    """(18)

    Args:
        wavelength (float): in m
        aperture (float): antenna aperture (m)
        alpha (float): Target angle in radian
        K: =1

    Returns (float): Beamwidth in radian.
    """
    return K * wavelength / aperture / np.cos(alpha)


def probability_limit(potential_db, distance, rcs):
    """(20)
    
    Args:
        potential_db: in dB
        distance: in m
        rcs: in m^2

    Returns:

    """
    potential_pow = db2pow(potential_db)
    return potential_pow * rcs / np.power(distance, 4)


def probability_detect_by_antenna_param(potential_db, distance, rcs, probability_alarm,
                                        num_pulse=1, k=1):
    """(21)

    Args:
        potential_db: in dB
        distance: in m
        rcs: in m^2
        probability_alarm:
        num_pulse:
        k:

    Returns:

    """
    limit = probability_limit(potential_db, distance, rcs)
    power = 1 / (np.power(limit, 2) * np.power(num_pulse, k) + 1)
    probability_detect = np.power(probability_alarm, power)
    return probability_detect


def probability_detect_by_range(distance, distance_max):
    """(22)

    Args:
        distance: in m
        distance_max: in m

    Returns:

    """
    return np.exp(-np.power(distance / distance_max, 4))


def std_angle(wavelength, aperture, alpha, snr_db, K=0.6):
    """(23, 24)

    Args:
        wavelength: in m
        aperture: in m
        alpha: in radian
        snr_db: in dB
        K:

    Returns:

    """
    beam_width = beam_width_by_aperture(wavelength, aperture, alpha, K)
    snr_pow = db2pow(snr_db)
    return beam_width / np.sqrt(snr_pow)


def std_distance(snr_db, tau=None, width_spec=None, tau_disc_fcm=None, mode='nonmod',
                 K=0.6):
    """(27-29)

    Args:
        snr_db: in dB
        tau: in seconds
        width_spec: in Hz
        tau_disc_fcm: in seconds
        mode ('nonmod', 'lfm', 'fcm'): Default nonmode
        K:

    Returns:

    """
    snr_pow = db2pow(snr_db)
    sigma = None
    if mode == 'nonmod':
        if tau is not None:
            sigma = K * scc.c * tau / np.sqrt(snr_pow)
    elif mode == 'lfm':
        if width_spec is not None:
            sigma = K * scc.c / width_spec / np.sqrt(snr_pow)
    elif mode == 'fcm':
        if tau_disc_fcm is not None:
            sigma = K * scc.c * tau_disc_fcm / np.sqrt(snr_pow)
    return sigma


def std_speed(snr_db=0, wavelength=None, tau=None, width_spec=None, T_lfm=None,
              mode='nonmod',
              K=0.6):
    """(27-29)

    Args:
        mode ('nonmod', 'lfm', 'fcm'): Default 'nonmode'
        snr_db: in dB
        wavelength: in m
        tau: in seconds
        width_spec: in Hz
        T_lfm: in seconds
        K:

    Returns:

    """
    snr_pow = db2pow(snr_db)
    sigma_speed = None
    if mode == 'nonmod' or mode == 'fcm':
        if tau is not None:
            sigma_speed = K * wavelength / tau / np.sqrt(snr_pow)
    elif mode == 'lfm':
        if width_spec is not None and T_lfm is not None:
            sigma_speed = np.sqrt(2) * K * scc.c / width_spec / np.sqrt(snr_pow) / T_lfm
    return sigma_speed


def std_h(distance, elevation, elevation_rls, snr_db, tau, aperture, wavelength,
          width_spec=None, tau_disc_fcm=None, mode='nonmode'):
    """(31)

    Args:
        mode ('nonmod', 'lfm', 'fcm'): Default 'nonmode'
        distance: in m
        elevation: in radian
        elevation_rls: in radian
        snr_db: in dB
        tau: in seconds
        aperture: in m
        wavelength: in m
        width_spec: in Hz
        tau_disc_fcm: in seconds

    Returns:

    """
    std_el = std_angle(wavelength, aperture, elevation, snr_db, K=0.6)
    if mode == 'nonmod':
        std_dist = std_distance(snr_db, tau=tau, mode='nonmod')
    elif mode == 'lfm':
        std_dist = std_distance(snr_db, width_spec=width_spec, mode='lfm')
    elif mode == 'fcm':
        std_dist = std_distance(snr_db, tau_disc_fcm=tau_disc_fcm, mode='lfm')
    else:
        return None
    std_dist_angle = [(np.sin(elevation_rls) + distance / EARTH_RADIUS) * std_dist,
                      distance * np.cos(elevation_rls) * std_el]
    return np.linalg.norm(std_dist_angle)


def std_x(distance, elevation, elevation_rls, azimuth, azimuth_rls,
          std_dist, std_el, std_az):
    """
    (32)

    Args:
        distance: in m
        elevation:  in radian
        elevation_rls: in radian
        azimuth: in radian
        azimuth_rls: in radian
        std_dist: in m
        std_el: in radian
        std_az: in radian

    Returns:

    """
    std_dist_angle = [
        std_dist * np.cos(elevation + elevation_rls) * np.cos(azimuth + azimuth_rls),
        std_az * distance * np.cos(elevation + elevation_rls) * np.sin(
            azimuth + azimuth_rls),
        std_el * distance * np.sin(elevation + elevation_rls) * np.cos(
            azimuth + azimuth_rls)
    ]
    return np.linalg.norm(std_dist_angle)


def std_y(distance, elevation, elevation_rls, azimuth, azimuth_rls,
          std_dist, std_el, std_az):
    """
    (32)

    Args:
        distance: in m
        elevation:  in radian
        elevation_rls: in radian
        azimuth: in radian
        azimuth_rls: in radian
        std_dist: in m
        std_el: in radian
        std_az: in radian

    Returns:

    """
    std_dist_angle = [
        std_dist * np.cos(elevation + elevation_rls) * np.sin(azimuth + azimuth_rls),
        std_az * distance * np.cos(elevation + elevation_rls) * np.cos(
            azimuth + azimuth_rls),
        std_el * distance * np.sin(elevation + elevation_rls) * np.sin(
            azimuth + azimuth_rls)
    ]
    return np.linalg.norm(std_dist_angle)


def std_xy(std_x, std_y):
    """(34)
    Args:
        distance:
        elevation:
        elevation_rls:
        azimuth:
        azimuth_rls:
        snr_db:
        aperture_h:
        aperture_w:
        wavelength:
        tau:
        width_spec:
        tau_disc_fcm:
        mode:

    Returns:

    """
    _std_x_y = [std_x, std_y]
    return np.linalg.norm(_std_x_y)


def delta_distance(width_spec):
    """(37)

    Args:
        width_spec: in Hz

    Returns:

    """
    return 0.5 * scc.c / width_spec


def delta_angle(wavelength, aperture, alpha=0, K=0.6):
    """(37)

    Args:
        wavelength: in m
        aperture: in m
        alpha: in radian
        K:

    Returns:

    """
    return beam_width_by_aperture(wavelength, aperture, alpha, K)


def delta_speed(wavelength, width_spec):
    """(37)

    Args:
        wavelength: in m
        width_spec: in Hz

    Returns:

    """
    return 0.5 * wavelength * width_spec


def delta_h(distance, elevation, elevation_rls, wavelength, aperture, K=0.6):
    """(37)

    Args:
        distance: in m
        elevation: in radian
        elevation_rls: in radian
        wavelength: in m
        aperture: in m
        K:

    Returns:

    """
    beal_el = beam_width_by_aperture(wavelength, aperture, elevation, K)
    return beal_el * distance / np.cos(elevation_rls + elevation)
