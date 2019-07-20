# distutils: language = c++

# cython: embedsignature = True
"""
Spherical extension module docstring
"""

import numpy as np
import cython
from cython.parallel import prange

from spharpy.samplings import Coordinates

cimport numpy as cnp
cimport libc.math as cmath
cimport libc.stdlib as cstdlib
cimport spharpy.special._special as _special

cdef extern from "math.h":
    double complex pow(double complex arg, double power) nogil

cdef extern from "<complex.h>" namespace "std":
    double complex exp(double complex z) nogil

cdef extern from "boost/math/special_functions/spherical_harmonic.hpp" namespace "boost::math":
    double complex spherical_harmonic(unsigned order, int degree, double theta, double phi) nogil;
    double spherical_harmonic_r(unsigned order, int degree, double theta, double phi) nogil;
    double spherical_harmonic_i(unsigned order, int degree, double theta, double phi) nogil;


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis_gradient(int n_max, coords):
    """
    TODO: correct docstring. This just copy and paste from SH basis
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [2]_.

    .. math::

        Y_n^m(\\theta, \\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-m)!}{(n+m)!}} P_n^m(\\cos \\theta) e^{i m \\phi}

    References
    ----------
    .. [2]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix
    """
    cdef cnp.ndarray[double, ndim=1] elevation
    cdef cnp.ndarray[double, ndim=1] azimuth

    if coords.elevation.ndim < 1:
        elevation = coords.elevation[np.newaxis]
        azimuth = coords.azimuth[np.newaxis]
    else:
        elevation = coords.elevation
        azimuth = coords.azimuth

    cdef Py_ssize_t n_points = elevation.shape[0]
    cdef Py_ssize_t n_coeff = (n_max+1)**2
    cdef cnp.ndarray[complex, ndim=2] grad_theta = \
        np.zeros((n_points, n_coeff), dtype=np.complex)
    cdef cnp.ndarray[complex, ndim=2] grad_phi = \
        np.zeros((n_points, n_coeff), dtype=np.complex)
    cdef complex[:, ::1] memview_theta = grad_theta
    cdef complex[:, ::1] memview_phi = grad_phi

    cdef double[::1] memview_azi = azimuth
    cdef double[::1] memview_ele = elevation

    cdef int point, acn, order, degree
    for point in range(0, n_points):
        for acn in prange(0, n_coeff, nogil=True):
            order = <int>(cmath.ceil(cmath.sqrt(<double>acn + 1.0)) - 1)
            degree = acn - order**2 - order

            memview_theta[point, acn] = \
                spherical_harmonic_function_derivative_theta( \
                order, degree, memview_ele[point], memview_azi[point])
            memview_phi[point, acn] = \
                    spherical_harmonic_function_gradient_phi( \
                order, degree, memview_ele[point], memview_azi[point])

    return grad_theta, grad_phi


def spherical_harmonic_normalization_full(int n, int m):
    if m>n:
        factor = 0
    else:
        factor = _special.tgamma_delta_ratio((n-m+1), (2*m))
        factor *= (2*n+1)/(4*np.pi)
        if m != 0:
            factor *= 2
        factor = cmath.sqrt(factor)
    return factor


def spherical_harmonic_function_derivative_theta_real(
        unsigned n,
        int m,
        double theta,
        double phi
        ):
    m_abs = np.abs(m)
    if n == 0:
        res = 0
    else:

        first = (n+m_abs)*(n-np.abs(m+1)) * _special.legendre_p(n, np.abs(m-1), np.cos(theta)) * (-1)**(m-1)
        second = _special.legendre_p(n, m_abs+1, np.cos(theta)) * (-1)**(m+1)
        legendre_diff = 0.5*(first - second)

        N_nm = spherical_harmonic_normalization_full(n, m_abs)

        if m<0:
            phi_term = np.sin(m_abs*phi)
        else:
            phi_term = np.cos(m_abs*phi)

        res = N_nm * legendre_diff * phi_term

    return res


def spherical_harmonic_function_grad_phi_real(
        int n,
        int m,
        double theta,
        double phi
        ):
    m_abs = np.abs(m)
    if m == 0:
        res = 0
    else:
        first = (n+m_abs)*(n+np.abs(m-1)) * _special.legendre_p(n-1, np.abs(m-1), np.cos(theta)) * (-1)**(m-1)
        second = _special.legendre_p(n-1, m_abs+1, np.cos(theta)) * (-1)**(m+1)
        legendre_diff = 0.5*(first + second)
        N_nm = spherical_harmonic_normalization_full(n, m_abs)

        if m<0:
            phi_term = np.cos(m_abs*phi)
        else:
            phi_term = -np.sin(m_abs*phi)

        res = N_nm * legendre_diff * phi_term

    return res


def spherical_harmonic_function_derivative_phi( \
        int n, int m, double theta, double phi):
    """Calculate the derivative of the spherical hamonics with respect to
    the azimuth angle phi.

    Parameters
    ----------

    n : int
        Spherical harmonic order
    m : int
        Spherical harmonic degree
    theta : double
        Elevation angle 0 < theta < pi
    phi : double
        Azimuth angle 0 < phi < 2*pi

    Returns
    -------

    sh_diff : complex double
        Spherical harmonic derivative

    """
    cdef double complex res
    if m == 0 or n == 0:
        res = 0.0
    else:
        res = spherical_harmonic_function(n, m, theta, phi) * 1j * m

    return res


cdef double complex spherical_harmonic_function_gradient_phi( \
        int n, int m, double theta, double phi) nogil:
    """Calculate the derivative of the spherical hamonics with respect to
    the azimuth angle phi divided by sin(theta)

    Parameters
    ----------

    n : int
        Spherical harmonic order
    m : int
        Spherical harmonic degree
    theta : double
        Elevation angle 0 < theta < pi
    phi : double
        Azimuth angle 0 < phi < 2*pi

    Returns
    -------

    sh_diff : complex double
        Spherical harmonic derivative

    """
    cdef double complex factor
    cdef double complex exp_phi, first, second, res, Ynm_sin_theta
    if m == 0:
        res = 0.0
    else:
        factor = cmath.sqrt((2*<double>n+1)/(2*<double>n-1))/2
        exp_phi = exp(1j*phi)
        first = cmath.sqrt((n+m)*(n+m-1)) * exp_phi * \
                spherical_harmonic_function(n-1, m-1, theta, phi)
        second = cmath.sqrt((n-m) * (n-m-1)) / exp_phi * \
                spherical_harmonic_function(n-1, m+1, theta, phi)
        Ynm_sin_theta = (-1) * factor * (first + second)
        res = Ynm_sin_theta * 1j

    return res



cdef double complex spherical_harmonic_function_derivative_theta(
        int n, int m, double theta, double phi) nogil:
    """Calculate the derivative of the spherical hamonics with respect to
    the elevation angle theta.

    Parameters
    ----------

    n : int
        Spherical harmonic order
    m : int
        Spherical harmonic degree
    theta : double
        Elevation angle 0 < theta < pi
    phi : double
        Azimuth angle 0 < phi < 2*pi

    Returns
    -------

    sh_diff : complex double
        Spherical harmonic derivative

    Note
    ----

    This implementation is subject to singularities at the poles due to the
    1/sin(theta) term.

    """
    cdef double complex exp_phi, first, second, res
    if n == 0:
        res = 0.0
    else:
        exp_phi = exp(1j*phi)
        first = cmath.sqrt((n-m+1) * (n+m)) * exp_phi * \
                spherical_harmonic_function(n, m-1, theta, phi)
        second = cmath.sqrt((n-m) * (n+m+1)) / exp_phi * \
                spherical_harmonic_function(n, m+1, theta, phi)
        res = (first-second)/2 * (-1)

    return res


cdef complex spherical_harmonic_function(unsigned n, int m, double theta, double phi) nogil:
    """Simple wrapper function for the boost spherical harmonic function."""
    cdef complex res
    if cstdlib.abs(m) > n:
        res = 0
    else:
        res = spherical_harmonic(n, m, theta, phi)
    return res


cdef double spherical_harmonic_function_real(unsigned n, int m, double theta, double phi) nogil:
    """Use c math library here for speed and numerical robustness.
    Using the numpy ** operator instead of the libc pow function yields
    numeric issues which result in sign errors."""

    cdef double Y_nm = 0.0
    if (m == 0):
        Y_nm = spherical_harmonic_r(n, m, theta, phi)
    elif (m > 0):
        Y_nm = spherical_harmonic_r(n, m, theta, phi) * cmath.sqrt(2)
    elif (m < 0):
        Y_nm = spherical_harmonic_i(n, m, theta, phi) * cmath.sqrt(2) * \
                <double>cmath.pow(-1, m+1)

    return Y_nm * <double>cmath.pow(-1, m)


def nm2acn(n, m):
    """
    Calculate the linear index coefficient for a spherical harmonic order n
    and degree m, according to the Ambisonics Channel Convention [1]_.

    .. math::

        acn = n^2 + n + m

    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    Returns
    -------
    acn : integer, ndarray
        Linear index

    """
    n = np.asarray(n, dtype=np.int)
    m = np.asarray(m, dtype=np.int)
    n_acn = m.size

    if not (n.size == m.size):
        raise ValueError("n and m need to be of the same size")

    acn = n**2 + n + m

    return acn


def acn2nm(acn):
    """
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the Ambisonics Channel Convention [1]_.

    .. math::

        n = \\lfloor \\sqrt{acn + 1} \\rfloor - 1

        m = acn - n^2 -n


    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    Returns
    -------
    acn : integer, ndarray
        Linear index

    """
    acn = np.asarray(acn, dtype=np.int)

    n = (np.ceil(np.sqrt(acn + 1)) - 1)
    m = acn - n**2 - n

    n = n.astype(np.int, copy=False)
    m = m.astype(np.int, copy=False)

    return n, m


cdef int acn2n(int acn) nogil:
    """ACN to n conversion with c speed and without global interpreter lock.
    """
    cdef int n
    n = <int>cmath.ceil(cmath.sqrt(<double>acn + 1)) - 1

cdef int acn2m(int acn) nogil:
    """ACN to m conversion with c speed and without global interpreter lock.
    """
    cdef int n = acn2n(acn)
    cdef int m = acn - <int>cmath.pow(n, 2) - n
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis(int n_max, coords):
    """
    Calulcates the complex valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [2]_, [3]_.

    .. math::

        Y_n^m(\\theta, \\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-m)!}{(n+m)!}} P_n^m(\\cos \\theta) e^{i m \\phi}

    References
    ----------
    .. [2]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [3]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8. Springer, 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Complex spherical harmonic basis matrix
    """
    cdef cnp.ndarray[double, ndim=1] elevation
    cdef cnp.ndarray[double, ndim=1] azimuth

    if coords.elevation.ndim < 1:
        elevation = coords.elevation[np.newaxis]
        azimuth = coords.azimuth[np.newaxis]
    else:
        elevation = coords.elevation
        azimuth = coords.azimuth

    cdef Py_ssize_t n_points = elevation.shape[0]
    cdef Py_ssize_t n_coeff = (n_max+1)**2
    cdef cnp.ndarray[complex, ndim=2] basis = \
        np.zeros((n_points, n_coeff), dtype=np.complex)
    cdef complex[:, ::1] memview_basis = basis

    cdef double[::1] memview_azi = azimuth
    cdef double[::1] memview_ele = elevation

    cdef int point, acn, order, degree
    for point in range(0, n_points):
        for acn in prange(0, n_coeff, nogil=True):
            order = <int>(cmath.ceil(cmath.sqrt(<double>acn + 1.0)) - 1)
            degree = acn - order**2 - order

            memview_basis[point, acn] = spherical_harmonic_function( \
                order, degree, memview_ele[point], memview_azi[point])

    return basis


@cython.boundscheck(False)
@cython.wraparound(False)
def spherical_harmonic_basis_real(int n_max, coords):
    """
    Calulcates the real valued spherical harmonic basis matrix of order Nmax
    for a set of points given by their elevation and azimuth angles.
    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [1]_.

    .. math::

        Y_n^m(\\theta, \\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\\cos \\theta)
        \\begin{cases}
            \displaystyle \\cos(|m|\\phi),  & \\text{if $m \\ge 0$} \\newline
            \displaystyle \\sin(|m|\\phi) ,  & \\text{if $m < 0$}
        \\end{cases}

    References
    ----------
    .. [1]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A Suggested Ambisonics
            Format (revised by F. Zotter),” International Symposium on Ambisonics and Spherical
            Acoustics, vol. 3, pp. 1–11, 2011.


    Parameters
    ----------
    n : integer
        Spherical harmonic order
    coordinates : Coordinates
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    Y : double, ndarray, matrix
        Real valued spherical harmonic basis matrix


    """
    if coords.elevation.ndim < 1:
        elevation = coords.elevation[np.newaxis]
        azimuth = coords.azimuth[np.newaxis]
    else:
        elevation = coords.elevation
        azimuth = coords.azimuth

    cdef Py_ssize_t n_points = elevation.shape[0]
    cdef Py_ssize_t n_coeff = (n_max+1)**2
    cdef cnp.ndarray[double, ndim=2] basis = \
        np.zeros((n_points, n_coeff), dtype=np.double)
    cdef double[:, ::1] memview_basis = basis

    cdef double[::1] memview_azi = azimuth
    cdef double[::1] memview_ele = elevation

    cdef int point, acn, order, degree
    for point in range(0, n_points):
        for acn in prange(0, n_coeff, nogil=True):
            order = <int>(cmath.ceil(cmath.sqrt(<double>acn + 1.0)) - 1)
            degree = acn - order**2 - order

            memview_basis[point, acn] = spherical_harmonic_function_real( \
                order, degree, memview_ele[point], memview_azi[point])

    return basis


@cython.boundscheck(False)
@cython.wraparound(False)
def modal_strength(int n_max,
                   cnp.ndarray[double, ndim=1] kr,
                   arraytype='rigid'):
    """
    Modal strenght function for microphone arrays.

    .. math::

        b(kr) =
        \\begin{cases}
            \displaystyle 4\\pi i^n j_n(kr),  & \\text{open} \\newline
            \displaystyle  4\\pi i^{(n-1)} \\frac{1}{(kr)^2 h_n^\\prime(kr)},  & \\text{rigid} \\newline
            \displaystyle  4\\pi i^n (j_n(kr) - i j_n^\\prime(kr)),  & \\text{cardioid}
        \\end{cases}


    Notes
    -----
    This implementation uses the second order Hankel function, see [4]_ for an
    overview of the corresponding sign conventions.

    References
    ----------
    .. [4]  V. Tourbabin and B. Rafaely, “On the Consistent Use of Space and Time
            Conventions in Array Processing,” vol. 101, pp. 470–473, 2015.


    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    kr : double, ndarray
        Wave number * radius
    arraytype : string
        Array configuration. Can be a microphones mounted on a rigid sphere,
        on a virtual open sphere or cardioid microphones on an open sphere.

    Returns
    -------
    B : double, ndarray
        Modal strenght diagonal matrix

    """
    arraytypes = {'open': 0, 'rigid': 1, 'cardioid': 2}
    cdef int config = arraytypes.get(arraytype)
    cdef int n_coeff = (n_max+1)**2
    cdef int n_bins = <int>kr.shape[0]

    cdef cnp.ndarray[complex, ndim=3] modal_strength = \
        np.zeros((n_bins, n_coeff, n_coeff), dtype=np.complex)
    cdef complex[:, :, ::1] mv_modal_strength = modal_strength

    cdef double[::1] mv_kr = kr

    cdef int n, m, acn
    cdef complex bn


    for k in range(0, n_bins):
        for n in prange(0, n_max+1, nogil=True):
            bn = _modal_strength(n, mv_kr[k], config)
            for m in range(-n, n+1):
                acn = n*n + n + m
                mv_modal_strength[k, acn, acn] = bn

    return np.squeeze(modal_strength)


cdef complex _modal_strength(int n, double kr, int config) nogil:
    """Helper function for the calculation of the modal strength for
    plane waves"""
    cdef complex modal_strength
    if config == 0:
        modal_strength = 4*cmath.pi*pow(1.0j, n) * _special.sph_bessel(n, kr)
    elif config == 1:
        modal_strength = 4*cmath.pi*pow(1.0j, n-1) / \
                _special.sph_hankel_2_prime(n, kr) / kr / kr
    elif config == 2:
        modal_strength = 4*cmath.pi*pow(1.0j, n) * \
                (_special.sph_bessel(n, kr) - 1.0j * _special.sph_bessel_prime(n, kr))

    return modal_strength


@cython.boundscheck(False)
@cython.wraparound(False)
def aperture_vibrating_spherical_cap(int n_max,
                           double rad_sphere,
                           double rad_cap):
    """
    Aperture function for a vibrating cap with radius :math:`r_c` in a rigid
    sphere with radius :math:`r_s` [5]_, [6]_

    .. math::

        a_n (r_{s}, \\alpha) =
        \\begin{cases}
            \displaystyle \\cos\\left(\\alpha\\right) P_n\\left[ \\cos\\left(\\alpha\\right) \\right] - P_{n-1}\\left[ \\cos\\left(\\alpha\\right) \\right],  & {n>0} \\newline
            \displaystyle  1 - \\cos(\\alpha),  & {n=0}
        \\end{cases}

    where :math:`\\alpha = \\arcsin \\left(\\frac{r_c}{r_s} \\right)` is the
    aperture angle.


    References
    ----------
    .. [5]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [6]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
            loudspeaker system as multipole source,” in Proceedings of the 33rd
            DAGA German Annual Conference on Acoustics, 2007, pp. 221–222.


    Parameters
    ----------
    n_max : integer, ndarray
        Maximal spherical harmonic order
    r_sphere : double, ndarray
        Radius of the sphere
    r_cap : double
        Radius of the vibrating cap

    Returns
    -------
    A : double, ndarray
        Aperture function in diagonal matrix form with shape
        :math:`[(n_{max}+1)^2~\\times~(n_{max}+1)^2]`

    """
    cdef double angle_cap = np.arcsin(rad_cap / rad_sphere)
    cdef double arg = np.cos(angle_cap)
    cdef int n_sh = (n_max+1)**2

    cdef double legendre_plus, legendre_minus

    cdef cnp.ndarray[double, ndim=2] aperture = \
            np.zeros((n_sh, n_sh), dtype=np.double)
    cdef double[:, ::1] mv_aperture = aperture

    aperture[0,0] = (1-arg)*2*np.pi**2
    cdef int n, m
    for n in range(1, n_max+1):
        legendre_minus = _special.legendre_p(n-1, arg)
        legendre_plus = _special.legendre_p(n+1, arg)
        for m in range(-n, n+1):
            acn = nm2acn(n, m)
            aperture[acn, acn] = (legendre_minus - legendre_plus) * \
                    4 * np.pi**2 / (2*n+1)

    return aperture


@cython.boundscheck(False)
@cython.wraparound(False)
def radiation_from_sphere(int n_max,
                          double rad_sphere,
                          cnp.ndarray[double, ndim=1] k,
                          double distance,
                          desity_medium=1.2,
                          speed_of_sound=343.0):
    """
    Radiation function in SH for a vibrating sphere including the radiation
    impedance and the propagation to a arbitrary distance from the sphere.


    TODO: This function does not have a test yet.


    References
    ----------
    .. [7]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [8]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
            loudspeaker system as multipole source,” in Proceedings of the 33rd
            DAGA German Annual Conference on Acoustics, 2007, pp. 221–222.


    Parameters
    ----------
    n_max : integer, ndarray
        Maximal spherical harmonic order
    r_sphere : double, ndarray
        Radius of the sphere
    k : double, ndarray
        Wave number
    distance : double
        Distance from the origin
    density_medium : double
        Density of the medium surrounding the sphere. Default is 1.2 for air.
    speed_of_sound : double
        Speed of sound in m/s

    Returns
    -------
    R : double, ndarray
        Radiation function in diagonal matrix form with shape
        :math:`[K \\times (n_{max}+1)^2~\\times~(n_{max}+1)^2]`



    """
    cdef int n_sh = (n_max+1)**2

    cdef double rho = desity_medium
    cdef double c = speed_of_sound
    cdef complex hankel, hankel_prime, radiation_order
    cdef int n_bins = <int>k.shape[0]
    cdef cnp.ndarray[complex, ndim=3] radiation = \
            np.zeros((n_bins, n_sh, n_sh), dtype=np.complex)
    cdef complex[:, :, ::1] mv_radiation = radiation

    cdef double[::1] mv_k = k

    cdef int n, m, kk
    for kk in range(0, n_bins):
        for n in range(0, n_max+1):
            hankel = _special.sph_hankel_2(n, mv_k[kk]*distance)
            hankel_prime = _special.sph_hankel_2_prime(n, mv_k[kk]*rad_sphere)
            radiation_order = hankel/hankel_prime * 1j * rho * c
            for m in range(-n, n+1):
                acn = nm2acn(n, m)
                radiation[kk, acn, acn] = radiation_order

    return radiation

