import numpy as np
import pandas as pd
from lmfit import fit_report, minimize, Parameters


def _cosinor(x, params):
    r'''1-harmonic cosine function'''

    A = params['Amplitude']
    phi = params['Acrophase']
    T = params['Period']
    M = params['Mesor']

    return M + A*np.cos(2*np.pi/T*x+phi)


def _residual(params, x, data, fit_func):
    r'''Residual function to minimize'''

    model = fit_func(x, params)
    return (data-model)


class Cosinor:
    """
    Class for Cosinor analysis.

    Cornelissen, G. (2014). Cosinor-based rhythmometry.
    Theoretical Biology and Medical Modelling, 11(1), 16.
    https://doi.org/10.1186/1742-4682-11-16

    """

    def __init__(
        self,
        fit_params=None
    ):

        self.__fit_func = _cosinor  # Fit function
        self.__fit_obj_func = _residual

        if fit_params is None:
            fit_params = Parameters()
            # Default parameters for the cosinor fit function
            fit_params.add('Amplitude', value=50, min=0)
            fit_params.add('Acrophase', value=np.pi, min=0, max=2*np.pi)
            fit_params.add('Period', value=1440, min=0)  # Dummy value
            fit_params.add('Mesor', value=50, min=0)
        self.__fit_initial_params = fit_params

    @staticmethod
    def _convert_timestamp_to_index(ts):
        r'''Convert timestamps'''
        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = ((ts.index - ts.index[0])/ts.index.freq).values
        return x

    @property
    def fit_func(self):
        r'''Cosinor fit function'''
        return self.__fit_func

    @property
    def fit_initial_params(self):
        r'''Initial parameters of the cosinor fit function'''
        return self.__fit_initial_params

    @fit_initial_params.setter
    def fit_initial_params(self, params):
        self.__fit_initial_params = params

    def fit(
        self,
        ts,
        params=None,
        method='leastsq',
        nan_policy='raise',
        reduce_fcn=None,
        verbose=False
    ):
        r'''Fit the actigraphy data using a cosinor function.

        Parameters
        ----------
        ts : pandas.Series
            Input time series.
        params: instance of Parameters [1]_, optional.
            Initial fit parameters. If None, use the default parameters.
            Default is None.
        method: str, optional
            Name of the fitting method to use [1]_.
            Default is 'leastsq'.
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:

            * 'raise': a ValueError is raised
            * 'propagate': the values returned from userfcn are un-altered
            * 'omit': non-finite values are filtered

            Default is 'raise'.
        reduce_fcn: str, optional
            Function to convert a residual array to a scalar value for the
            scalar minimizers. Optional values are:

            * 'None' : sum of squares of residual
            * 'negentropy' : neg entropy, using normal distribution
            * 'neglogcauchy': neg log likelihood, using Cauchy distribution

            Default is None.
        verbose: bool, optional
            If set to True, display fit informations.
            Default is False.

        Returns
        -------
        fit_results : MinimizerResult
            Fit results.

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html

        '''

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = self._convert_timestamp_to_index(ts)

        # Minimize residuals
        fit_results = minimize(
            self.__fit_obj_func,
            self.fit_initial_params if params is None else params,
            method=method,
            args=(x,  ts.values, self.fit_func),
            nan_policy=nan_policy,
            reduce_fcn=reduce_fcn
        )
        # Print fit parameters if verbose
        if verbose:
            print(fit_report(fit_results))

        return fit_results

    def best_fit(self, ts, params):
        """Best fit function of the data.

        Parameters
        ----------
        ts : pandas.Series
            Originally fitted time series.
        params: instance of Parameters [1]_
            Best fit parameters.

        Returns
        -------
        bestfit_data : pandas.Series
            Time series of the best fit data.

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html

        """

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = self._convert_timestamp_to_index(ts)
        y = self.fit_func(x, params)

        return pd.Series(index=ts.index, data=y)