# -*- coding: utf-8 -*-

import enum

import numpy as np
from numpy import float64
from scipy import linalg
from scipy import special as ssp


@enum.unique
class FluctuationModel(enum.Enum):
    Swerling0 = 0
    Swerling1 = 1
    Swerling2 = 2
    Swerling3 = 3
    Swerling4 = 4
    Swerling5 = 5


@enum.unique
class AngUnit(enum.Enum):
    deg = 0
    rad = 1


@enum.unique
class MagnitudeUnit(enum.Enum):
    linear_ratio = 0
    pow_ratio = 1
    dbi = 2


@enum.unique
class SamplesSource(enum.Enum):
    property_ = 0
    input_port_ = 1


class InvalidAzimuth(Exception):
    pass


class InvalidElevation(Exception):
    pass


def _cartesian_to_spherical(vector):
    """
    © http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
    Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.

    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """
    x = vector[0]
    y = vector[1]
    z = vector[2]
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return [az, elev, r]
    # print "====================== _cartesian_to_spherical ==========================="
    # res = np.zeros_like(vector)
    # # The radial distance.
    # # print vector[:, 0]
    # for i in range(vector.shape[1]):
    #     r = norm(vector[:, i])
    #     # print i, r
    #     if r == 0:
    #         continue
    #
    #     # Unit vector.
    #     unit = vector[:, i] / r
    #
    #     # The polar angle.
    #     theta = asin(unit[2])
    #     # The azimuth.
    #     phi = atan2(unit[1], unit[0])
    #     res[:, i] = np.asarray([r, theta, phi], dtype=float64)
    # print "==================================================================="
    # print "vector", vector
    # print "res", res
    # Return the spherical coordinate vector.
    # return res


def _spherical_to_cartesian(az, elev, r):
    z = r * np.sin(elev)
    rcosel = r * np.cos(elev)
    x = rcosel * np.cos(az)
    y = rcosel * np.sin(az)

    return [x, y, z]


def _spherical_to_cartesian_normal_order(r, az, elev):

    # az *= np.pi / 180
    # elev *= np.pi / 180
    z = r * np.sin(elev)
    rcosel = r * np.cos(elev)
    x = rcosel * np.cos(az)
    y = rcosel * np.sin(az)
    return [x, y, z]


def _cartesian_to_spherical_normal_order(vector):
    """
    © http://svn.gna.org/svn/relax/1.3/maths_fns/coord_transform.py
    Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.

    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """
    x = vector[0]
    y = vector[1]
    z = vector[2]
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    # elev = (np.arctan2(z, hypotxy) * 180 / np.pi) % 360
    elev = np.arctan2(z, hypotxy)
    # az = (np.arctan2(y, x) * 180 / np.pi) % 360
    az = np.arctan2(y, x)
    return [r, az, elev]


def rotazel(startaxes, azel, ang_units=AngUnit.deg):
    azel = convert_ang_units(np.asarray([azel[0], azel[1]], dtype=float64),
                             from_units=ang_units, to_units=AngUnit.rad)
    if len(azel) != 2:
        raise ValueError("Wrong azel value!")
    # rotate in the direction of x->y
    u = np.cos(azel[0])
    v = np.sin(azel[0])
    rotmat_z = np.asarray([[u, -v, 0], [v, u, 0], [0, 0, 1]], dtype=float64)
    # rotate in the direction of x->z
    u = np.cos(azel[1])
    w = np.sin(azel[1])
    rotmat_y = np.asarray([[u, 0, -w], [0, 1, 0], [w, 0, u]], dtype=float64)

    rotaxes = np.dot(rotmat_z, np.dot(rotmat_y, startaxes))
    return rotaxes


def incident2azel(inc_ang, refaxes, ang_units=AngUnit.deg):
    """
    Convert incident angle to az/el format
    """
    inc_ang = convert_ang_units(inc_ang, from_units=ang_units, to_units=AngUnit.rad)
    vector = np.asarray([np.ones(inc_ang.shape[1]), inc_ang[1], inc_ang[0]])

    lcl_orig = np.asarray([[0], [0], [0]])
    lcl_axes = refaxes
    lcl_coord = global2localcoord(vector, lcl_orig, lcl_axes)
    azel = np.asarray([lcl_coord[2, :], lcl_coord[1, :]])
    azel = convert_ang_units(azel, from_units=AngUnit.rad, to_units=ang_units)
    # print "azel", azel
    return azel


def global2localcoord(g_coord, local_orig, local_axes):

    g_coord = np.asarray(g_coord, dtype=float64)
    if g_coord.shape == (3, ):
        g_coord = np.asarray(g_coord)[:, np.newaxis]

    local_orig = np.asarray(local_orig, dtype=float64)
    if local_orig.shape == (3,):
        local_orig = np.asarray(local_orig)[:, np.newaxis]

    local_axes = np.asarray(local_axes, dtype=float64)
    # TODO: its only applicable for sph-to-sph transformation
    option = "ss"
    # if option[0] == 'r':
    #     pass
    # elif option[0] == 's':
    #     g_rect = np.zeros_like(g_coord)
    #     [g_rect[0], g_rect[1], g_rect[2]] = \
    #         _spherical_to_cartesian(g_coord[0], g_coord[1], g_coord[2])
    # else:
    #     pass

    lcl_rect = np.dot(np.transpose(local_axes), (g_coord - local_orig * np.ones((1, g_coord.shape[1]))))

    if option[1] == 'r':
        pass
    elif option[1] == 's':
        lcl_coord = np.zeros_like(lcl_rect)
        [lcl_coord[0], lcl_coord[1], lcl_coord[2]] = \
            _cartesian_to_spherical(lcl_rect)
    else:
        pass
    azel = np.asarray([lcl_coord[0, :], lcl_coord[1, :]])
    lcl_coord[0, :], lcl_coord[1, :] = convert_ang_units(azel, from_units=AngUnit.rad, to_units=AngUnit.deg)
    # # print "====================== global2localcoord ==========================="
    # # print "g_coord", g_coord
    # # print "local_orig", local_orig
    # # print "local_axes"
    # # print local_axes
    # # print "local_axes.transpose()"
    # # print local_axes.transpose()
    # g_rect = _spherical_to_cartesian(g_coord)
    # # print "g_rect", g_rect
    # lcl_rect = np.dot(local_axes, (g_rect - local_orig))
    # # print "lcl_rect", lcl_rect
    # lcl_coord = _cartesian_to_spherical(lcl_rect)
    # # print "lcl_coord", lcl_coord
    # # print "==================================================================="
    return lcl_coord


def rangeangle(pos, ref_pos=np.asarray([[0], [0], [0]]), ang_units=AngUnit.deg):
    """
    Args:
        pos(array-like): ``3-by-M`` matrix
        ref_pos(array-like): ``3-by-1`` matrix
        ang_units

    """
    pos = np.asarray(pos, dtype=np.float64)
    ref_pos = np.asarray(ref_pos, dtype=np.float64)

    assert len(pos.shape) == 2
    assert len(ref_pos.shape) == 2

    coords = pos - ref_pos

    ranges = np.linalg.norm(coords, axis=0)

    thetas = np.arccos(coords[2, :] / ranges)

    phis = np.arctan2(coords[1], coords[0])

    azimuths = phis
    elevations = np.pi / 2 - thetas

    if ang_units == AngUnit.deg:
        return ranges, np.asarray([azimuths, elevations]) * (180. / np.pi)
    elif ang_units == AngUnit.rad:
        return ranges, np.asarray([azimuths, elevations])
    else:
        raise ValueError("ang_units should be selected from AngularUnit enum")


def backbaffle(resp, angle):
    """
    Back baffle the response ``resp``.
    Note: there is no angle type and angle verification.
    It must be done by caller

    Args:
        resp(numpy.ndarray(float)):
            length-**M** vector the original response before the back baffling
        angle(numpy.ndarray(float)):
            is 2-by-**M** array whose columns are the angles where the corresponding responses in R are measured.
            Each column in ANG is in the form of [azimuth; elevation] (in degrees).
            g is a length-**M** array containing the response after the back baffling. After back baffling, responses
            corresponding to the angles that are beyond +/- 90 degrees in azimuth are set to 0.

            Note that the responses should be represented in linear scale.

            Set all azimuth beyond +/- 90 to 0.
            Note elevation +/- 90 should not be set to 0 even if azimuth is beyond
            +/- 90 because all elevation 90 is the same point. Need tolerance to deal with numerical round-offs.
    """
    abs = np.absolute
    # TODO introduce return g instead of resp modify?
    for i in range(resp.shape[0]):
        if (angle[0, i] < -90 - np.finfo(float).eps or angle[0, i] > 90 + np.finfo(float).eps) \
                and np.finfo(float).eps < abs(abs(angle[1, i]) - 90):
            resp[i] = 0
            # return resp


def pow2db(p):
    return 10 * np.log10(np.abs(p))


def db2pow(db):
    return np.power(10., 0.1 * db)


def mag2db(mag):

    return 20 * np.log10(np.abs(mag))


def db2mag(db):
    return np.power(10, (1. / 20.) * db)


def check_and_generate_angle_array(angle):
    angle = np.asarray(angle, dtype=np.float64)
    if len(angle.shape) == 0:  # Angle is a number
        angle_size = 1
        angle_array = np.zeros((2, angle_size,), dtype=np.float64)
        angle_array[0] = angle
    elif len(angle.shape) == 1:  # 1D-array
        angle_size = angle.shape[0]
        angle_array = np.zeros((2, angle_size,), dtype=np.float64)
        angle_array[0] = angle
    elif len(angle.shape) == 2:  # 2D-array with shape (2, n). n - number of angles
        if angle.shape[0] != 2:
            raise ValueError("Angle has more or less rows than 2.")
        else:
            angle_size = angle.shape[1]
            angle_array = angle
    else:
        raise ValueError("Angle has more dimensions than 2.")

    check_angle_values(angle_array)

    return angle_array


def check_angle_values(angle_array, ang_units=AngUnit.deg):  # TODO: add radian support
    assert ang_units == AngUnit.deg
    if (angle_array[0, :] < -180 - np.finfo(float).eps).any() or (angle_array[0, :] > 180 + np.finfo(float).eps).any():
        raise ValueError("wrong azimuth (beyond [-180;180])")
    if (angle_array[1, :] < -90 - np.finfo(float).eps).any() or (angle_array[1, :] > 90 + np.finfo(float).eps).any():
        raise ValueError("wrong elevation (beyond [-90;90])")


def freq_check_and_gen_array(freq):
    freq = np.asarray([freq]).squeeze()

    # freq.dtype != float and not np.issubdtype(freq.dtype, int):
    if not np.isscalar(freq) and not isinstance(freq, np.ndarray):
        raise TypeError("freq is broken. It contains `" + freq.dtype.__str__() + "` instead of scalar or ndarray")
    elif not np.issubdtype(freq.dtype, np.number):
        raise TypeError("freq is broken. It contains `" + freq.dtype.__str__() + "` instead of scalar or ndarray")
    elif len(freq.shape) > 1:
        raise ValueError("freq is not a vector")
    elif len(freq.shape) > 0:
        if freq.shape[0] == 0:
            raise ValueError("freq is empty")
        else:
            return freq
    else:
        return np.asarray([freq])


# TODO remove this from beamscan estimator
def unitarymat(m):
    """
    Unitary transformation matrix `Q`

    `Q = unitarymat(Hdoa,M)` returns a unitary transformation matrix `Q` that transforms a signal
    covariance matrix `Sx` with dimension `M x M` to a Real Symmetric Matrix `Sq`.

    Args:
        m(int):  dimensions
    """
    half = int(np.floor(m / 2.))
    i = np.eye(half)
    j = np.fliplr(i)

    if m % 2:
        # m odd
        u = np.matrix(np.zeros((half, 1,)))
        q_mid = np.concatenate((u.transpose(), np.asarray([[np.sqrt(2)]]), u.transpose()), axis=1)
        # print q_mid
        q1 = np.concatenate((i, u, 1j * i), axis=1)
        q3 = np.concatenate((j, u, -1j * j), axis=1)
        q = 1 / np.sqrt(2) * np.concatenate((q1, q_mid, q3), axis=0)
    else:
        # m even
        q1 = np.concatenate((i, 1j * i), axis=1)
        q3 = np.concatenate((j, -1j * j), axis=1)
        q = 1 / np.sqrt(2) * np.concatenate((q1, q3), axis=0)

    return np.matrix(q)


def convert_ang_units(angle, from_units=AngUnit.deg, to_units=AngUnit.rad):
    angle = np.asarray(angle, dtype=float64)

    if from_units == to_units:
        return angle
    elif from_units == AngUnit.deg and to_units == AngUnit.rad:
        return np.multiply(angle, np.pi / 180.)
    elif from_units == AngUnit.rad and to_units == AngUnit.deg:
        return np.multiply(angle, 180. / np.pi)
    else:
        raise ValueError("Error of misc import!")
    return angle


def systemp(nf, reftemp):
    return reftemp * db2pow(nf)


class PulseIntegration(enum.Enum):
    noncoherent = 0
    coherent_complex = 1
    coherent_real = 2


def findpeaks(y_axis, x_axis=None, min_peak_height=None):
    """
        Find local peaks in data
    """
    max_peaks = allpeakdetect(y_axis, x_axis)
    max_peaks_out = []

    if min_peak_height is not None:
        for (y, x) in max_peaks:
            if y >= min_peak_height:
                max_peaks_out.append([y, x])
    else:
        max_peaks_out = max_peaks

    return np.asarray(max_peaks_out)


def allpeakdetect(y_axis, x_axis=None, lookahead=1, delta=0):
    """
    Based on a MATLAB script at: http://billauer.co.il/peakdet.html function for detecting
    local maximas and minmias in a signal. Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas.

    Args:
        y_axis:
            A list containg the signal over which to find peaks

        x_axis:
            (optional) A x-axis whose values correspond to the y_axis list
            and is used in the return to specify the postion of the peaks. If
            omitted an index of the y_axis is used. (default: None)

        lookahead: (optional) distance to look ahead from a peak candidate to
            determine if it is the actual peak (default: 1)
            '(sample / period) / f' where '4 >= f >= 1.25' might be a good value

        delta: (optional) this specifies a minimum difference between a peak and
            the following points, before a peak may be considered a peak. Useful
            to hinder the function from picking up false peaks towards to end of
            the signal. To work well delta should be set to delta >= RMSnoise * 5.
            (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    Returns:
        [max_peaks, min_peaks]:
            lists containing the positive and negative peaks respectively.
            Each cell of the lists contains a tupple of: (position, peak_value)
            to get the average peak value do: np.mean(max_peaks, 0)[1] on the
            results to unpack one of the lists into x, y coordinates do:
                x, y = zip(*tab)
    """
    max_peaks = []

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis, lookahead, delta)
    # store data length for later use
    length = len(y_axis)

    # maximum candidate are temporarily stored
    mx = -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    # print "x_axis", x_axis
    # print "y_axis", y_axis
    # print x_axis[1:-lookahead]
    # print y_axis[1:-lookahead]
    # print list(enumerate(zip(x_axis[1:-1], y_axis[1:-1]), 1))
    for index, (x, y) in enumerate(zip(x_axis[1:], y_axis[1:]), 1):
        if y > mx:
            mx = y
            mxpos = x

        # print x, y_axis[index:index + lookahead], y_axis[index:index + lookahead].max(), "mx = ", mx

        if y < mx - delta and mx != np.Inf:
            # Maximum peak candidate found look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() <= mx:
                max_peaks.append([mx, mxpos])
                # set algorithm to only find minimum now
                mx = -np.Inf
                if index + lookahead > length:
                    # end is within lookahead no more peaks can be found
                    break
            continue

    return max_peaks


def _datacheck_peakdetect(x_axis, y_axis, lookahead, delta):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise (ValueError, 'Input vectors y_axis and x_axis must have same length')

    # needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)

    # perform some other checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    return x_axis, y_axis


def az2broadside(az, el, ang_units=AngUnit.deg):
    az = convert_ang_units(az, from_units=ang_units, to_units=AngUnit.rad)
    el = convert_ang_units(el, from_units=ang_units, to_units=AngUnit.rad)
    bsd = np.sin(az) * np.cos(el)
    bsd = np.clip(bsd, -1, 1)  # TODO use this func for back_buffle
    bsd = np.arcsin(bsd)
    bsd = convert_ang_units(bsd, from_units=AngUnit.rad, to_units=ang_units)
    return bsd


def broadside2az(bsd, el, ang_units=AngUnit.deg):
    bsd = convert_ang_units(bsd, from_units=ang_units, to_units=AngUnit.rad)
    el = convert_ang_units(el, from_units=ang_units, to_units=AngUnit.rad)
    az = np.sin(bsd) / np.cos(el)
    az = np.clip(az, -1, 1)  # TODO use this func for back_buffle
    az = np.arcsin(az)
    az = convert_ang_units(az, from_units=AngUnit.rad, to_units=ang_units)
    return az


def integrate_pulses(in_signal, axis, coherent=False):
    """
    If `coherent` : :math:`y_i = \\sum\\limits_{j=1}^N x_{ij}`.

    Else: :math:`y_i = \\sqrt{\\sum\\limits_{j=1}^N|x_{ij}|^2}`.

    Args:
        in_signal(array-like): Pulse input data as ``M-by-N`` matrix. ``M`` is the number of samples in each pulse,
            ``N`` is the number of pulses.
        axis(int): by each axis to integrate
        coherent(bool): how to integrate, coherent or not.
    Returns:
        out_signal(:class:`numpy.ndarray`): an 1D vector with length ``M``
    """
    in_signal = np.asarray(in_signal, dtype=np.complex128)
    if len(in_signal.shape) < 2:
        raise ValueError("in_signal should be a matrix or 3D array")
    if not isinstance(coherent, bool):
        raise ValueError("coherent should be bool")

    if coherent:
        out_signal = np.sum(in_signal, axis=axis)
    else:
        out_signal = np.sqrt(np.sum(np.square(np.abs(in_signal)), axis=axis))
    return out_signal


def _azel2vec(azel):
    """
    Compute the direction vector for azimuth and elevation
    [AZ_VEC, EL_VEC] = azelvec(AZEL) computes the
    increasing direction vectors, AZ_VEC and EL_VEC, at azimuth and
    elevation angles (in degrees) specified in AZEL.

    AZEL is a 2xN matrix whose columns are azimuth and elevation angle
    pairs in the form of [az;el]. Azimuth angles must be between -180 and
    180 while elevation angles must be between -90 and 90. AZ_VEC and
    EL_VEC are 3xN matrix whose columns represent the corresponding
    direction vectors in azimuth and elevation directions, respectively.
    The vectors are in the form of [x;y;z].

    Args:
        azel(array-like): azimuth and elevation angles
    Returns:
        out(tuple(array-like)): tuple of increasing direction vectors, az_vec and el_vec
    """
    if len(azel.shape) > 1:
        theta = np.radians(azel[1, :])
        phi = np.radians(azel[0, :])
    else:
        theta = np.radians(azel[1])
        phi = np.radians(azel[0])
    el_vec = np.asarray([-np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), np.cos(theta)])
    az_vec = np.asarray([-np.sin(phi), np.cos(phi), np.zeros(phi.size)])
    return az_vec, el_vec


def _azelcoord(azimuth, elevation):
    """
    Coordinate system at azimuth and elevation direction
    M = azelcoord(AZ,EL) returns the axes of coordinate
    system, (az_hat, el_hat, r_hat), at the direction of (AZ,EL) (in
    degrees) in the (x,y,z) system where az and el are defined. M is a 3x3
    matrix whose columns represent the three axes, az_hat, el_hat, and
    r_hat. AZ must be between -180 and 180. El must be between -90 and 90.

    The coordinate system is constructed at the direction of (AZ,EL). The
    three axes are az_hat, the direction of increasing azimuth; el_hat, the
    direction of increasing elevation; and r_hat, the increasing radial
    direction.

    Args:
        azimuth(float): azimuth angle in degrees
        elevation(float): elevation angle in degrees
    Returns:
        M(array-like): axes of coordinate system at the direction (azimuth, elevation)
    """
    az_vec, el_vec = _azel2vec(np.asarray([azimuth, elevation]))
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    r_vec = np.asarray([np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth),
                        np.sin(elevation)])
    return np.swapaxes(np.asarray([az_vec, el_vec, r_vec]), 0, 1)


def azelaxes(azimuth, elevation):
    """
    Axes at given azimuth and elevation direction
    AX = azelaxes(AZ,EL) returns the components of the set of three
    orthonormal basis vectors at a point on the unit sphere. The basis
    vectors are unit vectors in the radial, azimuthal, and elevation
    directions. The point on the sphere is specified by the azimuth angle
    AZ (in degrees) and the elevation angle EL (in degrees).

    AX is a 3x3 matrix whose columns contain the unit vectors in the
    radial, azimuthal, and elevation directions, respectively.

    Args:
        azimuth(float): azimuth angle in degrees
        elevation(float): elevation angle in degrees

    Returns:
        AX(array-like): 3x3 matrix whose columns contain the unit vectors in the radial, azimuthal,
                        and elevation directions, respectively
    """
    if azimuth > 180 or azimuth < -180:
        raise InvalidAzimuth("Azimuth angle must be >= -180 and <= 180 degrees")
    if elevation > 90 or elevation < -90:
        raise InvalidElevation("Elevation angle must be >= -90 and <= 90 degrees")
    return _azelcoord(azimuth, elevation)[:, [2, 0, 1]]


def spherical_to_cartesian(vec, azimuth, elevation):
    m = _azelcoord(azimuth, elevation)
    return np.dot(m.squeeze(), vec)


def local_to_global_vec(lcl_vec, lcl_axes):
    return np.dot(lcl_axes, lcl_vec)


def cartesian_to_spherical_vec(vec, azimuth, elevation):
    m = _azelcoord(azimuth, elevation)
    return np.dot(m.T, vec)


def global_to_local_vec(g_vec, lcl_axes):
    return np.dot(lcl_axes.T, g_vec)


def create_rectangular_grid(num_elements, size, spacing):
    positions = np.zeros(shape=(num_elements, 3), dtype=np.float64)

    z_length = (size[0] - 1) * spacing[0]
    y_length = (size[1] - 1) * spacing[1]

    n = 0
    for j in range(size[1]):
        for i in range(size[0]):
            positions[n] = np.asarray([0, -y_length / 2 + j * spacing[1], z_length / 2 - i * spacing[0]],
                                      dtype=np.float64)
            n += 1
    return positions


def _qrlinsolve(lhs, rhs, return_f=False):
    """
    Solves linear equations (A*A')X = B using QR decomposition, where A is rhs and B is lhs
    Args:
        lhs(array-like): matrix A
        rhs(array-like): matrix B
        return_f(bool): return intermediate solution
    Returns:
        x(array-like): solution of the equation
        f(array-like): intermediate solution during decomposition (only of return_f=True)
    """

    q, r, pivotperm = linalg.qr(np.conj(lhs.T), pivoting=True, mode='economic')

    if _matlab_matrix_division:
        f = matrix_divide.mldivide(np.asfortranarray(np.conj(r.T)), np.asfortranarray(rhs[pivotperm, :]))
        x = matrix_divide.mldivide(np.asfortranarray(r), np.asfortranarray(f))
    else:
        if r.shape[0] == r.shape[1]:
            f = linalg.solve_triangular(np.conj(r.T), rhs[pivotperm, :], lower=True)
            x = linalg.solve_triangular(r, f, lower=False)
        else:
            f = matrix_divide.mldivide(np.conj(r.T), rhs[pivotperm, :])
            x = matrix_divide.mldivide(r, f)

    pivotpermsrt = np.argsort(pivotperm)
    x = x[pivotpermsrt, :]
    if return_f:
        return x, f[pivotpermsrt, :]
    else:
        return x


def lcmvweights(data, constraints, desired_responses, diagonal_loading):
    """
    Calculates the linear constraint minimum variance (LCMV) weights using sample matrix
    inversion (SMI) method.
    Args:
        data(array-like): MxN data matrix whose rows are snapshots in time.
            constraints(array-like): NxL matrix whose columns are constraints and L stands
            for the number of constraints.
        desired_responses(array-like): length L column vector whose elements are desired
            responses corresponding to the constraints specified in constraints.
        diagonal_loading(float): the diagonal loading factor for the data covariance matrix.
            Must be non-negative.
    Returns:
        w(array-like): length N column vector representing the LCMV weights.
    """

    desired_responses = np.asarray(desired_responses)
    desired_responses = np.reshape(desired_responses, (desired_responses.size, 1))
    data_dim_m, data_dim_n = data.shape
    x = data / np.sqrt(data_dim_m)
    delta = np.sqrt(diagonal_loading)
    if delta != 0:
        x = np.append(x, delta * np.identity(data_dim_n))
    if constraints.shape[1] is 1:
        # MVDR
        temp = _qrlinsolve(x.T, constraints)
        w = (desired_responses * temp) / (np.dot(np.conj(constraints.T), temp))
    else:
        # LCMV
        if data_dim_m >= data_dim_n:
            temp, f = _qrlinsolve(x.T, constraints, return_f=True)
            w = temp * _qrlinsolve(np.conj(f.T), desired_responses)
        else:
            # when matrix is fat, f is no longer square and we cannot play the trick on thin matrix. Therefore, we
            # have to form R2 and use LU decomposition
            temp = _qrlinsolve(x.T, constraints)
            r2 = np.dot(np.conj(constraints.T), temp)
            l2, u2 = linalg.lu(r2, permute_l=True)
            temp2 = linalg.solve(u2, linalg.solve(l2, desired_responses))
            w = temp * temp2
    return w


def mrdivide(b, a):
    """
    The operators / and \ are related to each other by the equation B/A = (A'\B')'.
    """

    b = to_dtype(b, np.complex128, 2)
    a = to_dtype(a, np.complex128, 2)
    x = np.conj(matrix_divide.mldivide(np.conj(a).T, np.conj(b).T)).T
    return x


def lcmvweights_new_style(constr, response, cov):
    """
    Caulculates narrowband linearly constrained minimum variance (LCMV) beamformer weights
    Args:
        constr (array-like): a complex-valued matrix with shape ``N_sensors``-by-``K_constraints``
        response (array-like): a complex-valued 1D vector, which size is equal to ``K_constraints``
        cov (array-like): sensor signal spatial covariation matrix, shaped ``N_sensors``-by-``N_sensors``.
            It includes signal from all incoming signals and all noise.
    """

    constr = to_dtype(constr, np.complex128, 2)
    n_sens, k_constr = constr.shape
    response = to_dtype(response, np.complex128, 1, {0: k_constr})
    cov = to_dtype(cov, np.complex128, 2, {0: n_sens, 1: n_sens})

    # TODO: ensure scov_in is Hermitian + check for positive semi definite

    a_s = constr
    cov_inv_as = matrix_divide.mldivide(cov, a_s)
    ash_covinv_as = np.dot(np.conj(np.transpose(a_s)), cov_inv_as)

    w = np.dot(mrdivide(cov_inv_as, (ash_covinv_as + np.conj(ash_covinv_as).T) / 2.), response)

    return w


def elemdelay(pos, c, ang, ang_units=AngUnit.deg):
    """
    Delay among sensor elements in a sensor array

    tau = elemdelay(POS,C,ANG) returns the delay among sensor elements in a sensor array for a given direction
    specified in ang.

    Args:
        pos: is a 3-row matrix representing the positions of the elements.
            Each column of POS is in the form of [x;y;z] (in meters).

        c: is a scalar representing the propagation speed (in m/s).

        ang: is a 2-row matrix representing the incident directions. Each column of ANG is in the form
            of [azimuth;elevation] form (in ang_units). Azimuth angles must be within
            [-180 180] and elevation angles must be within [-90 90].

    Returns:
    tau: is an NxM matrix where N is the number of columns in `pos` and M is the number of columns in `ang`.
    """

    ang = convert_ang_units(ang, from_units=ang_units, to_units=AngUnit.rad)  # TODO redesign angles!!!
    azang = ang[0, :]
    elang = ang[1, :]

    # angles defined in local coordinate system
    incidentdir = np.asarray([-np.cos(elang) * np.cos(azang),
                              -np.cos(elang) * np.sin(azang),
                              -np.sin(elang)])

    tau = np.dot(pos.transpose(), incidentdir) / float(c)
    return tau


def steeringvec(pos, freq, c, ang):
    tau = elemdelay(pos, c, ang)

    freq = np.asarray([freq]).squeeze(0)
    sv = np.exp(-1j * 2 * np.pi * freq * tau)
    return sv


# TODO: docstring
# FIXME: get rid of noise_gen_seed
def sensorsig(pos, n_snapshots, ang, ncov_in=0, scov_in=1, noise_gen_seed=0, taper=None):
    """
    This function creates signals from angles `angs` with random initial phase and collects them to sensors.

    Parameters
    ----------
    pos : `array-like`
        bla-bla

    n_snapshots: `int`
        bla-bla`
    ang: `array-like`
        bla
    ncov_in: `scalar or array-like`
        bla
    scov_in: `scalar or array-like`
        bla
    noise_gen_seed: `int`
        bla
    taper: `scalar or array-like`
        bla

    Returns
    -------
    x: :class:`numpy.ndarray`
        received signal, a ``n_samples``-by-``N_sensors`` matrix
    rxx_theoretical: :class:`numpy.ndarray`
        theoretical signal covariance matrix, shaped ``N_sensors``-by-``N_sensors``
    rxx: :class:`numpy.ndarray`
        observed signal covariance matrix, shaped ``N_sensors``-by-``N_sensors``
    """

    ang = check_and_generate_angle_array(ang)
    pos = to_dtype(pos, dtype=np.float64)
    noise_gen_seed = int(noise_gen_seed)

    n_snapshots = int(n_snapshots)

    n_elem = pos.shape[1]
    n_ang = ang.shape[1]

    if taper is not None:
        taper = to_dtype(taper, np.complex128, 1, {0: n_elem})

    noise_flag = True
    if np.isscalar(ncov_in):
        ncov = ncov_in * np.eye(n_elem)
        if np.isclose(ncov_in, 0):
            noise_flag = False
    elif len(ncov_in.shape) is 1:
        ncov = np.diag(ncov_in)
    else:
        # TODO add check if cond = any(any(abs(ncov_in - ncov_in') > tol));
        ncov = ncov_in

    if np.isscalar(scov_in):
        scov = scov_in * np.eye(n_ang)
    elif len(scov_in.shape) is 1:
        scov = np.diag(scov_in)
    else:
        # TODO add check if cond = any(any(abs(scov_in - scov_in') > tol));
        scov = scov_in

    if np.isscalar(scov):
        su = np.sqrt(scov)
        su = np.asarray([su])
    else:
        scov_chol = scov
        try:
            suv = linalg.cholesky(scov_chol, lower=False)
        except linalg.LinAlgError:
            sed, sev = linalg.eigh((scov + scov.conjugate().transpose()) / 2.)  # FIXME not same result in matlab
            tol = np.spacing(1)
            sed[np.abs(sed) < tol] = np.abs(sed[np.abs(sed) < tol])
            sed[np.abs(sed) < tol / 10] = 0  # FIXME not same result in matlab
            cond = sed < 0
            if any(cond):
                raise AssertionError("notPositiveSemiDefinite SCOV in misc.sensorsig")
            posidx = sed > 0
            sut = np.dot(sev[:, posidx], np.diag(np.sqrt(sed[posidx])))
            su = sut.transpose().conjugate()
        else:
            suf = suv[0:n_ang, 0:n_ang]
            su = suf

    noise_gen = noise.NoiseSource(distribution=RandomDistribution.uniform,  # FIXME set mode = vsl
                                  seed=noise_gen_seed, mode=Mode.matlab)
    phase_noise = noise_gen.step(1, (n_snapshots, su.shape[0],))
    x_in = np.dot(np.exp(1j * 2 * np.pi * phase_noise), su)
    sv = steeringvec(pos, 1, 1, ang)
    if taper is not None:
        sv *= taper[:, None]
    x = np.dot(x_in, sv.transpose())
    # print x.shape

    if noise_flag:
        ncov_chol = ncov
        nu = linalg.cholesky(ncov_chol, lower=False)
        noise_gen.distribution = RandomDistribution.gaussian
        noise_gen.output_complex = True

        noise_sig = noise_gen.step(1, (n_snapshots, n_elem,))
        noise_sig = np.dot(noise_sig, nu)
        x_out = x + noise_sig
    else:
        x_out = x

    # theoretical covariance matrix Rxx_theor
    rxx_cov_the = np.dot(sv, np.dot(scov, np.conj(sv.T))) + ncov
    rxx_cov_the = (rxx_cov_the + np.conj(rxx_cov_the.T)) / 2.  # ensure Hermitian

    # and signal covariance matrix Rxx
    rxx_cov = np.dot(x_out.transpose(), np.conj(x_out)) / n_snapshots
    rxx_cov = (rxx_cov + np.conj(rxx_cov.transpose())) / 2.  # ensure Hermitian

    return x_out, rxx_cov_the, rxx_cov


def to_dtype(source_array, dtype, n_dims=None, dims_to_check={}):
    """
    Args:
        source_array(array-like):
        dtype(numpy.numeric): to what `dtype` to convert
        n_dims(int): how much dimensions should be in this array (don't check if `None`)
        dims_to_check(dict): dict with dimensions to check {what_dimension_to_check: what_should_it_be}
    """

    target_array = np.asarray(source_array, dtype=dtype)
    if n_dims is not None:
        if len(target_array.shape) != n_dims:
            raise ValueError("Source array should have %i dimensions (now %i)" % (n_dims, len(target_array.shape)))
    for dim, val in dims_to_check.items():
        if target_array.shape[dim] != val:
            raise ValueError("Source array should have %i elements in %i-th dimension" % (val, dim))
    return target_array


def dop2speed(freq_shift, wavelength):
    """
    Args:
        freq_shift(array-like)
        wavelength(float)
    Returns:
        v_dop(float): doppler speed estimation
    """

    freq_shift = np.asarray(freq_shift, dtype=np.float64)
    wavelength = float(wavelength)
    return freq_shift * wavelength


def speed2dop(rad_vel, wavelength):
    """
    :math:`\\Delta f = \\frac{v_{rad}}{\\lambda}`

    Args:
        rad_vel(array-like): :math:`v_{rad}`, radial speeds in m/s (approaching is positive)
        wavelength(float): :math:`\\lambda`, wavelength in meters
    """

    rad_vel = np.asarray(rad_vel, dtype=np.float64)
    wavelength = float(wavelength)
    return rad_vel / wavelength


if __name__ == "__main__":
    ang = np.asarray([[30, 40], [0, 0]])
    # print ang
    pos = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    npower = 0.01
    sensorsig(pos, 2, ang, ncov_in=npower)


def find_nearest(point, grid_list, last_element=None, circle=False):
    grid = np.abs(np.array(grid_list) - point)
    nearest = np.argmin(grid)
    if circle is True:
        if nearest == len(grid_list) - 1:
            if np.abs(grid_list[nearest] - point) > np.abs(last_element - point):
                return 0

    else:
        if last_element is not None and point >= last_element:
            return None

    return nearest


def find_nearest2(point, grid_list, last_element=None, circle=False):
    idx = np.searchsorted(grid_list, point, side="left")
    if idx > 0 and (idx == len(grid_list) or np.abs(point - grid_list[idx - 1]) < np.abs(point - grid_list[idx])):
        nearest = idx - 1
    else:
        nearest = idx

    if circle is True:
        if nearest == len(grid_list) - 1:
            if np.abs(grid_list[nearest] - point) > np.abs(last_element - point):
                return 0
    else:
        if last_element is not None and point >= last_element:
            return None
    return nearest


def find_nearest3(point, grid_list, last_element=None, circle=False):

    for el in range(len(grid_list)):
        if point < grid_list[el]:
            break
    idx = el
    if idx > 0 and (idx == len(grid_list) or np.abs(point - grid_list[idx - 1]) < np.abs(point - grid_list[idx])):
        nearest = idx - 1
    else:
        nearest = idx

    if circle is True:
        if nearest == len(grid_list) - 1:
            if np.abs(grid_list[nearest] - point) > np.abs(last_element - point):
                return 0
    else:
        if last_element is not None and point >= last_element:
            return None
    return nearest


def marcumq(a, b):
    N = 1
    X = np.power(a, 2) / 2 / N
    Y = np.power(b, 2) / 2
    Q = np.zeros(X.shape)
    for idx in range(len(X)):
        Q[idx] = probability_of_detection_by_shnidman(N, X[idx], Y)
    return Q


def probability_of_detection_by_shnidman(N, X, Y):
    # Computest thet probability of detection as given by Shnidman[1].

    Q = None
    NX = N * X  # Notational convenience

    # Evaluate the Chernoff bound
    if Y == 0:
        logCB = 0  # Avoid division by zero
    else:
        Lambda = 1 - N / (2 * Y) - np.sqrt(np.power(N / (2 * Y), 2) + NX / Y)  # Eq. 22
        logCB = -Lambda * Y + NX * Lambda / (1 - Lambda) - N * np.log(
            1 - Lambda)  # Eq. 23

    if np.exp(logCB) < np.finfo(float).tiny:
        if Y < NX + N:
            Q = 1
        elif Y > NX + N:
            Q = 0
    else:
        if Y < NX + N:  # Compute 1-P(N, X, Y) using form 4
            Form = 4  # Form 4 corresponds to Eq. 11
        else:  # Compute P(N, X, Y) using form 2
            Form = 2  # Form 2 corresponds to Eq. 9 end

        if Form == 2:
            # Form 2 corresponds to Eq. 9
            # Adjust start of inner summation to avoid underflow
            kmin = findStartOfSummation(NX, 0)

            # Adjust start of outer summation to avoid underflow
            mmin = findStartOfSummation(Y, kmin + N)

            # Initialize m and k to adjusted values
            m = mmin
            k = m - N

            # Initialize the values of innerSum and outerSum If value of k based on N and m is greater than the
            #  minimum value computed above, pre - compute start of inner summation.
            # If not, then we can ignore the first k - 1 terms in the inner summation.
            if kmin < k:
                innerSum = 0
                for idx in range(kmin, k + 1):
                    innerArg = expA(NX, idx)
                    if innerArg > np.finfo(float).tiny:
                        innerSum = innerSum + innerArg
            else:
                innerArg = expA(NX, k)
                innerSum = innerArg

            # With m = mmin, ignore the first m - 1 terms in the outer summation
            outerArg = expA(Y, m)
            outerSum = outerArg * (1 - innerSum)

            # Sum until the outerSum is no longer increasing
            while innerArg > np.finfo(float).eps and \
                    outerArg * (1 - innerSum) > np.finfo(float).tiny:
                m = m + 1
                k = k + 1
                innerArg = innerArg * NX / k
                innerSum = innerSum + innerArg
                outerArg = outerArg * Y / m
                outerSum = outerSum + outerArg * (1 - innerSum)

            # Form 2 also has a summation over m separate from the double sum term
            num_million = np.floor(N / 1e6).astype(int)
            for m in range(1, num_million + 1):
                outerSum = outerSum + np.sum(expA(Y, range((m - 1) * 1e6, m * 1e6)))

            outerSum = outerSum + np.sum(expA(Y, np.arange(num_million * 1e6, N)))

            Q = outerSum

        elif Form == 4:
            # Form 4 corresponds to Eq. 11 Adjust start of inner summation to avoid underflow
            mmin = findStartOfSummation(Y, 0)

            # Adjust start of outer summation to avoid underflow
            kmin = findStartOfSummation(NX, np.max(mmin - N + 1, 0))

            # Initialize k and m to adjusted values
            k = kmin
            m = N - 1 + k
            # Initialize the values of innerSum and outerSum
            # If value of m based on N and k is greater than the minimum value computed above,
            # pre - compute start of inner summation.If not, then we can ignore
            # the first m - 1 terms in the inner summation.
            if mmin < m:
                innerSum = 0
                for idx in np.arange(mmin, m + 1):
                    innerArg = expA(Y, idx)
                    if innerArg > np.finfo(float).tiny:
                        innerSum = innerSum + innerArg
            else:
                innerArg = expA(Y, m)
                innerSum = innerArg

            # With k = kmin, ignore the first k - 1 terms in the outer summation
            outerArg = expA(NX, k)
            outerSum = outerArg * (1 - innerSum)

            # Sum until the outerSum is no longer increasing
            while innerArg > np.finfo(float).eps and outerArg * (
                    1 - innerSum) > np.finfo(float).tiny:
                m = m + 1
                k = k + 1
                innerArg = innerArg * Y / m
                innerSum = innerSum + innerArg
                outerArg = outerArg * NX / k
                outerSum = outerSum + outerArg * (1 - innerSum)

            Q = 1 - outerSum
    return Q


def findStartOfSummation(constTerm, minVal):
    epsM = 1e-40

    G = -4 * np.log(4 * epsM * (1 - epsM)) / 5  # Eq. 41

    # the parameter constTerm should be large enough for Shnidman's assumptions
    # to hold. If the parameter is not large enough, the ensuing calculations
    # are meaningless (we get a square root with negative input in Eq. 40).
    # Shnidman only offers a solution to the underflow region problem for large
    # constTerm because:
    # 1. This is effectively the case of interest
    # 2. If constTerm is not large, the solution can't be obtained in a expedient manner, and is thus inefficient.
    # Here, we assume that if (2*constTerm - G)>0, the assumptions hold and the
    # underflow region is investigated. Else, this step is skipped.
    if expA(constTerm, minVal) > np.finfo(float).tiny or 2 * constTerm - G < 0:
        # If there is no underflow in the minVal term, or if simplifying
        # assumptions do not hold, start summation at minVal
        startIdx = minVal
    else:
        # Underflow is detected in the minVal term, so compute new starting index
        startIdx = np.floor(
            constTerm + 1 / 2 - np.sqrt(G * (2 * constTerm - G)))  # Eq. 40

    return startIdx


def expA(y, n):
    # Evaluates terms of the form exp(A) = (e^-y)*(y^n)/(n!).  For large values of y
    # we use a modified expression for exp(A) from a follow-up paper by Shnidman:
    #
    # [3] D. A. Shnidman, "Note on 'The Calculation of the Probability of
    # Detection and the Generalized Marcum Q-Function'", IEEE Transactions
    # on Information Theory, vol. 37, no. 4, p. 1233, July 1991.

    if y > 0:
        if y > 1e4:
            # Note:  There is a typo in the equation for "A" in [3], affecting the
            # first term: (z+0.5) should read (z-0.5)
            # and the term in the denominator of the first term in the brackets:
            # 1+1/(2z) should read 1-1/(2z).
            z = n + 1
            x = np.exp((z - 0.5) * ((1 - y / z) / (1 - 1. / (2 * z)) + np.log(y / z)) -
                       0.5 * np.log(2 * np.pi * y) - J(z))
        else:
            x = np.exp(-y + n * np.log(y) - ssp.gammaln(n + 1))
    else:
        # In this case y equals zero, so the answer is zero for all cases except
        # when n equals zero, in which case the answer is one
        if n == 0:
            x = 1
        else:
            x = 0
    return x


def J(z):
    # A continued fraction approximation to the Binet function given in [1]
    x = 1. / (12 * z + 2. / (5 * z + 53. / (42 * z + 1170. / (53 * z + 53. / z))))  # Eq.B5
    return x


def roty(beta):
    beta = convert_ang_units(beta, from_units=AngUnit.deg, to_units=AngUnit.rad)
    rotmat = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return rotmat


def rotx(beta):
    beta = convert_ang_units(beta, from_units=AngUnit.deg, to_units=AngUnit.rad)
    rotmat = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    return rotmat


def polloss(fv_t, fv_r, pos_r=None, ax_r=None, pos_t=None, ax_t=None):
    if ax_t is None:
        ax_t = np.eye(3)
    if pos_t is None:
        pos_t = np.asarray([0, 0, 0]).reshape(3, 1)
    if ax_r is None:
        ax_r = np.eye(3)
    if pos_r is None:
        pos_r = np.asarray([0, 0, 0]).reshape(3, 1)

    # TODO validate3DCartCoord

    rcoord_t = global2localcoord(pos_r, pos_t, ax_t)
    vec = list((fv_t / np.linalg.norm(fv_t)).ravel())
    vec.append(0.)
    fv_t_local = spherical_to_cartesian(vec, rcoord_t[0], rcoord_t[1])
    fv_t_global = local_to_global_vec(fv_t_local, ax_t)

    tcoord_r = global2localcoord(pos_t, pos_r, ax_r)
    vec = list((fv_r / np.linalg.norm(fv_r)).ravel())
    vec.append(0.)
    fv_r_local = spherical_to_cartesian(vec, tcoord_r[0], tcoord_r[1])
    fv_r_global = local_to_global_vec(fv_r_local, ax_r)
    rho = abs(np.dot(fv_t_global, fv_r_global)) ** 2
    rho = -pow2db(rho)
    return rho
