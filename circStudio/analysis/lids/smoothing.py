from math import sqrt,log
from scipy.ndimage import gaussian_filter1d

def spm_smooth(Y, fwhm=5.0):
    """
    Smooth 1D signals using a Gaussian filter kernel parameterized by
    full width at half maximum (FWHM).

    This function applied Gaussian smoothing using ``scipy.ndimage.gaussian_filter1d``. Unlike the standard
    SciPy interface, the width of the kernel is specified as full-width at
    half-maximum (FWHM) rather than standard deviation.

    Parameters
    ----------
    Y : ndarray, shape (J, Q)
        Array of 1D signals to be smoothed.
    fwhm : float, optional
        Full-width at half-maximum of the Gaussian kernel, in samples.
        Default is 5.0.

    Returns
    -------
    ndarray, shape (J, Q)
        Smoothed signals.

    References
    ----------
    This function is derived from code originally distributed as part of the spm1d package
    and is reused here in accordance with the GNU General Public License v3.0 (GPL-3.0).
    The function is included to avoid dependency conflicts associated with the full spm1d
    package.
    """
    sd = fwhm / sqrt(8 * log(2))
    return gaussian_filter1d(Y, sd, mode='wrap')