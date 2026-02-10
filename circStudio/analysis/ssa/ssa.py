import numpy as np
import pandas as pd
from scipy import linalg


class SSA:
    """
    Singular Spectrum Analysis (SSA)

    SSA is a PCA-like method for time series. It decomposes the actigraphic
    signal into additive components and estimates how important each component
    is.

    1) **Embed** the time series into a matrix where each column is a sliding
    windows of the signal (the *trajectory matrix*).
    2) **Decompose** the trajectory matrix into a set of ranked components using
    Singular Value Decomposition (SVD). Largest component explains most variance,
    like PCA.
    3) **Reconstruct** one component (or a group of them) back into a time series
    by diagonal averaging.

    Parameters
    ----------
    data : pandas.Series
        Time series with a DateTimeIndex and a defined sampling frequency
        (i.e., `data.index.freq` must not be None).
    window_length : str or pandas.Timedelta, optional
        Size of the embedding window. For example "24H" or "90min".
        This is converted into an integer number of samples based on the series
        sampling frequency.

    References
    ----------
    This code is derived from the original implementation in pyActigraphy, distributed under the BSD 3-Clause License.
    Original author: Grégory Hammad (gregory.hammad@uliege.be).

    [1] Hammad, G., Reyt, M., Beliy, N., Baillet, M., Deantoni, M., Lesoinne, A., Muto, V., & Schmidt, C. (2021).
    pyActigraphy: Open-source python package for actigraphy data visualization and analysis.
    PLoS Computational Biology, 17(10), 1009514–1009535. https://doi.org/10.1371/journal.pcbi.1009514

    [2] Hammad, G., Wulff, K., Skene, D. J., Münch, M., & Spitschan, M. (2024). Open-Source Python Module for the
    Analysis of Personalized Light Exposure Data from Wearable Light Loggers and Dosimeters.
    LEUKOS, 20(4), 380–389. https://doi.org/10.1080/15502724.2023.2296863

    [3] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum Analysis for Time Series. Springer Berlin
    Heidelberg. http://doi.org/10.1007/978-3-642-34913-3

    Examples
    --------
    >>> ssa = SSA(activity, window_length="24h")
    >>> ssa.fit()
    >>> ssa.variance_explained.sum()
    >>> trend = ssa.X_tilde(0)
    >>> reconstructed = ssa.reconstructed_signal([0,1,2,3,4,5,6])
    >>> w_corr_mat = ssa.w_correlation_matrix(10)
    """
    # -------------------------
    # Construction/validation
    # -------------------------
    def __init__(self, data: pd.Series, window_length: str|pd.Timedelta = '24h'):

        # ----------------------------
        # Input time series
        # ----------------------------

        # The input signal must be a pd.Series
        if not isinstance(data, pd.Series):
            raise TypeError("`data` must be a pandas.Series.")

        # SSA assumes regularly sampled data
        # Sampling frequency is used to convert time windows (e.g. 24h)
        # into a number of samples
        if data.index.freq is None:
            raise ValueError(
                "SSA requires a regularly sampled time series (data.index.freq is None).\n"
                "Fix: resample your series first, e.g. series.resample('1min').mean()."
            )

        # Store the original time series
        self.data = data

        # Store data sampling interval
        self.freq = pd.Timedelta(data.index.freq)

        # -----------------------------
        # Window / embedding parameters
        # -----------------------------

        # Window length defines the size of the sliding window
        # used to build the trajectory matrix.
        # Intuitively, a small window captures short-term structure
        # A large window captures long-term structure
        window_length = (pd.Timedelta(window_length)
              if not isinstance(window_length, pd.Timedelta)
              else window_length)

        # Number of samples (rows) in each column, that is, in
        # each sliding window. This  window length (L) is also
        # called "embedding dimension"
        self.L = int(pd.Timedelta(window_length)/self.freq)

        # Number of windows I can slide over the signal (K)
        # This determines how many columns the trajectory matrix has
        # If N is the length of the signal, K = N - L + 1
        self.K = len(data.values) - self.L + 1

        # ------------------------
        # Results filled by fit()
        # ------------------------

        # Left singular vectors => each column represents a basic pattern
        # within each window
        self.U = None

        # Singular values, measuring the importance of each component
        self.sigma = None

        # Right singular vectors (transposed)
        # Describe how each pattern evolves over time
        self.Vh = None

        # Fraction of total variance explained by each component
        self.variance_explained = None

    # ---------------
    # Core SSA steps
    # ---------------

    def trajectory_matrix(self):
        """
        Build the trajectory matrix (a Hankel matrix) from the time series.

        Each column corresponds to a sliding window of length `L` (number of rows)
        and `K` (number of columns), where :math:`K = N - L + 1`.
        """
        # Collect the time series (ts)
        ts = self.data.values

        # Collect columns and rows
        columns = ts[:self.L]
        rows = ts[-self.K:]

        # Return Hankel matrix
        return linalg.hankel(columns, rows)

    def fit(self, check_finite=False, overwrite_a=True):
        """
        Decompose trajectory matrix using singular value decomposition (SVD).

        In SSA, we first build a trajectory matrix (a Henkel matrix, where each
        column represents an overlapping window of the signal). SVD splits this
        matrix into three other matrices:

            - U: patterns that describe what the component looks like inside a window.
            - S: how much does each component explain structure/variance in the signal.
            - Vh: how each component changes across windows.

        For each component, `fit()` also computes how the fraction of total variance
        explained by each component.

        Parameters
        ----------
        check_finite : bool, optional
            If True, SciPy checks for NaN/inf values inside the trajectory matrix.
            Safer, but slower. Default is False.
        overwrite_a : bool, optional
            If True, SciPy may overwrite the trajectory matrix during the SVD to save memory.
            Default is True.
        """
        # Build the trajectory matrix (overlapping windows of the signal)
        a = self.trajectory_matrix()

        # Obtain U, S and Vh
        u, s, vh = linalg.svd(
            a, full_matrices=False, check_finite=check_finite, verwrite_a=overwrite_a
        )
        self.U = u
        self.sigma = np.diag(s)
        self.Vh = vh
        self.variance_explained = np.square(s)/np.sum(np.square(s))


    @staticmethod
    def _weights(L, K):

        N = L + K
        # weights = np.empty(N-1, dtype=np.float32)
        weights = np.empty(N - 1, dtype=np.int32)
        for k in range(1, L):
            weights[k - 1] = k

        # for k in range(L,K+1):
        weights[L - 1:K] = L

        for k in range(K + 1, N):
            weights[k - 1] = N - k

        return weights

    # ------------------------------------
    # Component similarity (W-correlation)
    # ------------------------------------
    @staticmethod
    def _weighted_scalar_product(X, Y, w):
        return np.dot(X, np.multiply(Y, w).T)

    def _weighted_correlation(self, X, Y, w):
        """
        Weighted correlation between two reconstructed components.

        In SSA, diagonal averaging gives edge points fewer contributions.
        W-correlation accounts for that so correlation isn’t biased by edges.
        """
        w_norm_X = np.sqrt(self.__class__._weighted_scalar_product(X, X, w))
        w_norm_Y = np.sqrt(self.__class__._weighted_scalar_product(Y, Y, w))

        w_rho = self.__class__._weighted_scalar_product(X, Y, w) / (w_norm_X * w_norm_Y)

        return w_rho

    @staticmethod
    def _x_elementary(U, s, Vh, L, K, i):

        X_i = np.empty((L, K), dtype=np.float32)

        # Implement the dot product s * U[,i] x Vh[i].T
        sVh_i = s * Vh[i]
        for j in range(L):
            X_i[j] = U[j, i] * sVh_i

        return X_i

    def _diagonal_averaging(self, X):

        L, K = X.shape
        L_star, K_star = min(L, K), max(L, K)
        # N_star = L_star + K_star
        if not L < K:
            X = X.T

        sum_antidiags = np.empty(L_star + K_star - 1, dtype=np.float32)
        for k in range(1 - L_star, K_star):
            # Avoid using np.flipud as it does not compile with numba.
            # Besides, it seems slower than [::-1,...]
            sum_antidiags[k + L_star - 1] = np.trace(X[::-1, ...], offset=k)

        scale_factors = self.__class__._weights(L_star, K_star)

        sum_antidiags /= scale_factors

        return sum_antidiags



    def X_elementary(self, r):
        r'''Elementary matrix

        Parameters
        ----------
        r: int
            Index of the elementary matrix.
            Must lower or equal to the embedding dimension, L.

        Returns
        -------
        x_elem: ndarray of shape (L,K)


        Notes
        -----

        The SVD of the trajectory matrix X can be written as [1]_ :

        .. math:

            X = X_1 + \ldots + X_R

        where :math:`X_r = \sqrt{\lambda_r} u_r v_{r}^\intercal`.

        The matrices :math:`X_r` have rank 1. Such matrices are sometimes
        called *elementary* matrices.
        '''
        #  TODO: check if r is in range

        X_r = self.__class__._x_elementary(
            self.__U,
            self.__sigma[r][r],
            self.__Vh,
            self.__L,
            self.__K,
            r
        )

        return X_r

    def X_tilde(self, r):
        r'''Diagonal averaged matrix.

        Parameters
        ----------
        r: int or list of int
            Index of the elementary matrix to be diagonal-averaged.
            Must be lower than or equal to the embedding dimension, L.
            If a list of indices is given instead, the corresponding elementary
            matrices are grouped (ie. reduced to a single matrix by summation)
            before diagonal-averaging.

        Returns
        -------
        x_tilde: ndarray of shape (M,)


        Notes
        -----

        [1]_ : if the components of the series are separable and the indices
        are being split accordingly, then all the matrices in the expansion
        :math:`X = X_{I_1} + \ldots + X_{I_m}` are the Hankel matrices.
        We thus immediately obtain the decomposition
        :math:`x_n = \sum_{k=1}^m \tilde{x}_n^{(k)}` of the original series:
        for all k and n, :math:`\tilde{x}_n^{(k)}` is equal to all entries
        :math:`x^{(k)}_{ij}` along the antidiagonal
        :math:`{(i, j)| i + j = n+1}` of the matrix :math:`X_{Ik}`. In
        practice, however, this situation is not realistic. In the general
        case, no antidiagonal consists of equal elements. We thus need a formal
        procedure of transforming an arbitrary matrix into a Hankel matrix and
        therefore into a series. As such, we shall consider the procedure of
        *diagonal averaging*, which defines the values of the time series

        .. math::

            \tilde{\mathbb{X}}^{(k)} = \left(
                \tilde{x}^{(k)}_1, \ldots, \tilde{x}^{(k)}_N \right)

        as averages for the corresponding antidiagonals of the matrices
        :math:`X_{I_k}`.

        * for :math:`1 \leq n < L^{\star}`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{n} *
             \sum_{m=1}^{n} x^{\star}_{I_k, (m,n-m+1)}

        * for :math:`L^{\star} \leq n < K^{\star}`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{L^{\star}} *
             \sum_{m=1}^{L^{\star}} x^{\star}_{I_k, (m,n-m+1)}

        * for :math:`K^{\star} < n \leq N`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{N-n+1} *
             \sum_{m=n-K^{\star}+1}^{N-K^{\star}+1} x^{\star}_{I_k, (m,n-m+1)}
        '''
        if isinstance(r, list):
            X_elementaries = [self.X_elementary(i) for i in r]
            from functools import reduce
            X_elementary = reduce((lambda x, y: np.add(x, y)), X_elementaries)
        else:
            X_elementary = self.X_elementary(r)

        X_tilde = self.__class__._diagonal_averaging(X_elementary)

        return X_tilde

    def reconstructed_signal(self, n):
        r'''Reconstructed signal from diagonal averaged matrices.

        Parameters
        ----------
        n: array of int
            Indices of the diagonal-averaged matrices to merge.
            Must be lower than or equal to the embedding dimension, L.

        Returns
        -------
        reco: pandas.Series

        '''

        X_tildes = [self.X_tilde(i) for i in n]

        # add the X_tilde matrices recursively
        from functools import reduce
        X_reco = reduce((lambda x, y: np.add(x, y)), X_tildes)

        reco_signal = pd.Series(
            data=X_reco,
            index=self.__data.index
        )

        return reco_signal

    def w_correlation_matrix(self, k):
        r'''W-correlation matrix.

        Parameters
        ----------
        k: int
            Maximal index of the diagonal-averaged matrices to use.
            Must be lower than or equal to the embedding dimension, L.

        Returns
        -------
        wmat: numpy.ndarray

        '''

        n = range(k)

        w_corr_mat = np.empty((k, k))

        w = self.__class__._weights(self.__L, self.__K)

        X_tildes = [self.X_tilde(i) for i in n]

        for i in n:
            for j in n[i:]:
                w_corr = self.__class__._weighted_correlation(
                    X_tildes[i],
                    X_tildes[j],
                    w
                )
                w_corr_mat[i][j] = w_corr
                w_corr_mat[j][i] = w_corr

        return w_corr_mat