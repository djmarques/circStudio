import numpy as np
import pandas as pd
from scipy import linalg
from functools import reduce


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
    def __init__(self, data: pd.Series, window_length: str | pd.Timedelta = "24h"):

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
        window_length = (
            pd.Timedelta(window_length)
            if not isinstance(window_length, pd.Timedelta)
            else window_length
        )

        # Number of samples (rows) in each column, that is, in
        # each sliding window. This  window length (L) is also
        # called "embedding dimension"
        self.L = int(pd.Timedelta(window_length) / self.freq)

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
    # Step 1: Embed
    # ---------------

    def build_trajectory_matrix(self) -> np.ndarray:
        """
        Build the trajectory matrix (a Hankel matrix) from the time series.

        Each column corresponds to a sliding window of length `L` (number of rows)
        and `K` (number of columns), where :math:`K = N - L + 1`.
        """
        # Collect sequence of values from data
        values = self.data.values

        # Extract number of columns
        columns = values[: self.L]

        # Extract number of rows
        rows = values[-self.K :]

        # Construct Hankel matrix with L columns and K rows
        return linalg.hankel(columns, rows)

    # -----------------------
    # Step 2: Decompose (SVD)
    # -----------------------

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
        trajectory = self.build_trajectory_matrix()

        # Compute SVD store U, s and Vh
        svd = linalg.svd(
            trajectory,
            full_matrices=False,
            check_finite=check_finite,
            overwrite_a=overwrite_a,
        )
        # Store U, s and Vh
        self.U = svd[0]
        self.sigma = np.diag(svd[1])
        self.Vh = svd[2]

        # Calculate fraction of variance explained by each component
        self.variance_explained = np.square(svd[1]) / np.sum(np.square(svd[1]))

    # --------------------------
    # Step 3: Reconstruct signal
    # --------------------------

    @staticmethod
    def _weights(num_rows: int, num_cols: int) -> np.ndarray:
        """
        Calculate how many elements belong to each anti-diagonal of a matrix.

        Anti-diagonals are sets of elements whose row and column indices sum to
        a constant.

        Example
        -------
        For the 3*3 Hankel matrix:

            [[1,2,3],
            [3,4,5],
            [4,5,6]]

        the anti-diagonals are:

            [[1],
            [2,2],
            [3,3,3],
            [4,4],
            [5]]

        Notice how:
            - Anti-diagonals near the edges contain fewer elements.
            - Anti-diagonals near the center contain more elements.

        In SSA, diagonal averaging is used to reconstruct a time series by
        averaging each anti-diagonal of a matrix.

        Because different anti-diagonals contain different numbers of elements,
        we must divide by their counts. Otherwise, central time points would
        receive more contributions than edge points, introducing bias.

        Parameters
        ----------
        num_rows : int
            Number of rows of the matrix (L).
        num_cols : int
            Number of columns of the matrix (K).

        Returns
        -------
        np.ndarray
            A 1D array where each entry contains the number of elements
            in the corresponding anti-diagonal.
        """
        # Calculate total number of anti-diagonals present in the matrix
        total_antidiagonals = num_rows + num_cols - 1

        # Create empty vector to store count of matrix elements in each anti-diagonal
        weights = np.empty(total_antidiagonals, dtype=np.int32)

        # Increasing part (the anti-diagonals grow)
        for k in range(1, num_rows):
            weights[k - 1] = k
        # Constant middle part (plateau in the size of the anti-diagonals)
        weights[num_rows - 1 : num_cols] = num_rows

        # Decreasing part (the anti-diagonals shrink)
        for k in range(num_cols + 1, total_antidiagonals + 1):
            weights[k - 1] = total_antidiagonals + 1 - k

        return weights

    def _diagonal_averaging(self, component_matrix: np.ndarray) -> np.ndarray:
        """
        Convert a trajectory matrix (a Hankel matrix) back to a 1D signal by
        averaging its anti-diagonals.

        In SSA reconstruction of the actigraphy signal, each time point of the
        reconstructed component is obtained by taking the mean of one anti-diagonal
        of the component matrix. Think about each component as a particular pattern
        that explains part of the variance in the signal.

        Steps
        -----
            - Sum each anti-diagonal of the component matrix.
            - Divide by how many elements that anti-diagonal contains (weights)

        Parameters
        ----------
        component_matrix : np.ndarray
            Matrix of shape (L, K) representing an SSA component in matrix form.

        Returns
        -------
        np.ndarray
            Reconstructed 1D signal of length L + K - 1 (same length as the original
            series).
        """
        # Check if component_matrix is a 2D array
        if component_matrix.ndim != 2:
            raise ValueError("The input matrix must be a 2D array.")

        # Unpack the number of rows and columns
        rows, cols = component_matrix.shape

        # The number of rows must be smaller than the number of columns
        # Transpose matrix if that is not the case
        if rows > cols:
            component_matrix = component_matrix.T
            rows, cols = component_matrix.shape

        # Total number of anti-diagonals in the matrix
        antidiagonals = rows + cols - 1

        # Initialize array to accumulate sums for each anti-diagonal
        sums = np.zeros(antidiagonals, dtype=float)

        # Accumulate values into their corresponding anti-diagonal (r + c)
        for r in range(rows):
            for c in range(cols):
                sums[r + c] += component_matrix[r, c]

        # Compute how many elements belong to each anti-diagonal
        counts = self.__class__._weights(rows, cols)

        # Return the mean value of each anti-diagonal
        return sums / counts

    def get_component_matrix(self, r: int) -> np.ndarray:
        """
        Return one SSA component in matrix form.

        Parameters
        ----------
        r : int
        Index of the SSA component (0 = most important component).

        Returns
        -------
        np.ndarray
            Matrix representation of the selected component.

        Notes
        -----
        You must call `fit()` before using this method.
        """
        # Check if user has already run fit() method
        if self.U is None or self.sigma is None or self.Vh is None:
            raise RuntimeError(" Run fit() before calling this method.")

        # Check if requested number of components (R) is valid
        if not (0 <= r < self.Vh.shape[0]):
            raise ValueError("Invalid row index.")

        # Rename L and K self-parameters to become clearer
        rows = self.L
        columns = self.K

        # Allocate the (rows, columns) matrix for the component
        component_matrix = np.empty((rows, columns), dtype=np.float32)

        # For a single component i, use X_r = s_r * u_r * v_r^T, where:
        #   - s_r is the singular value
        #   - u_r is the i-th left singular vector (column of U)
        #   - v_r^T is the i-th right singular vector (row of Vh)

        # Scale the i-th right singular vector (v_i) by its singular value (s_i)
        scaled_right_vector = self.sigma[r][r] * self.Vh[r]

        # For each row of the component matrix,
        for row in range(rows):
            component_matrix[row] = self.U[row, r] * scaled_right_vector

        return component_matrix


    def reconstruct_component(self, r):
        """
        Reconstruct one (or several grouped) SSA component(s) as a 1D time series.

        In SSA, each component is first represented as a matrix of shape (L, K).
        To convert that matrix back into a time series, we perform diagonal averaging:
        we average the values along each anti-diagonal of the matrix, which produces
        a reconstructed 1D signal of length L + K - 1.

        Parameters
        ----------
        r: int or list[int]
            Index (or indices) of the SSA component(s) to reconstruct.
            - If an int is given, reconstruct that single component (0 = strongest)
            - If a list/tuple of ints is given, those component matrices are summed
            first and then diagonal-averaged.

        Returns
        -------
        np.ndarray:
            A 1D array of length `L + K - 1` (same length as the original input series).

        Notes
        -----
        You must call `fit()` before using this method.
        """
        if isinstance(r, list):
            # If multiple component indices are provided, build each component matrix and sum them
            component_matrices = [self.get_component_matrix(i) for i in r]

            # Sum all selected component matrices into a single matrix
            component_matrices = reduce((lambda x, y: np.add(x, y)), component_matrices)
        else:
            # If a single component index is provided, retrieve its matrix representation directly
            component_matrices = self.get_component_matrix(r)

        # Convert the (L, K) component matrix back to a 1D signal by averaging its anti-diagonals
        return self._diagonal_averaging(component_matrices)


    def reconstruct_signal(self, n):
        """
        Reconstruct actigraphy signal by summing several SSA components

        This method reconstructs multiple SSA components (each as 1D time
        series via diagonal averaging) and then adds them together to produce
        a combined reconstruction.

        Typical use cases
        -----------------
        - Reconstruct the trend using the first component(s): [0] or [0, 1]
        - Reconstruct a smoothed signal using the first few components: [0..k]
        - Reconstruct a band of components that represent a rhythm or noise

        Parameters
        ----------
        n: list[int]
            Component indices to include in the reconstruction.
            Example: [0, 1, 2] sums the first three SSA components.


        Returns
        -------
        pandas.Series
            Reconstructed signal aligned to the original time index.
            Same length as the input series.
        """
        # Reconstruct each requested component into a 1D signal
        reconstructed_components = [self.reconstruct_component(i) for i in n]

        # Sum all reconstructed components into one combined signal
        reconstructed = reduce((lambda x, y: np.add(x, y)), reconstructed_components)

        # Wrap the reconstructed array back into a pandas.Series with the original time index
        return pd.Series(data=reconstructed, index=self.data.index)

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

    def w_correlation_matrix(self, k):
        r"""W-correlation matrix.

        Parameters
        ----------
        k: int
            Maximal index of the diagonal-averaged matrices to use.
            Must be lower than or equal to the embedding dimension, L.

        Returns
        -------
        wmat: numpy.ndarray

        """

        n = range(k)

        w_corr_mat = np.empty((k, k))

        w = self.__class__._weights(self.__L, self.__K)

        X_tildes = [self.X_tilde(i) for i in n]

        for i in n:
            for j in n[i:]:
                w_corr = self.__class__._weighted_correlation(
                    X_tildes[i], X_tildes[j], w
                )
                w_corr_mat[i][j] = w_corr
                w_corr_mat[j][i] = w_corr

        return w_corr_mat
