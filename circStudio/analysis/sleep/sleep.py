import numpy as np
import pandas as pd
import re
import warnings
from .scoring import csm, roenneberg, sleep_midpoint, sri
from .scoring.utils import rescore
from scipy.ndimage import binary_closing, binary_opening
from circStudio.analysis.tools import *
from circStudio.analysis.sleep.sleep_tools import *


def AonT(data, whs=12):
    r"""Activity onset time.

    Activity onset time derived from the daily activity profile.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    whs: int, optional
        Window half size.
        Default is 12.

    Returns
    -------
    aot: Timedelta
        Activity onset time.

    """
    dailyprof = _average_daily_activity(data, cyclic=False)
    return _activity_onset_time(dailyprof, whs=whs)


def AoffT(data, whs=12):
    r"""Activity offset time.

    Activity offset time derived from the daily activity profile.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    whs: int, optional
        Window half size.
        Default is 12.

    Returns
    -------
    aot: Timedelta
        Activity offset time.

    """
    dailyprof = _average_daily_activity(data, cyclic=False)
    return _activity_offset_time(dailyprof, whs=whs)


def Cole_Kripke(data, settings=None, threshold=1.0, rescoring=True):
    r"""Cole&Kripke algorithm for sleep-wake identification.

    Algorithm for automatic sleep scoring based on wrist activity,
    developped by Cole, Kripke et al [1]_.


    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    settings: str, optional
        Data reduction settings for which the optimal parameters have been
        derived. Available settings are:

        * "mean": mean activity per minute
        * "10sec_max_overlap": maximum 10-second overlapping epoch per
          minute
        * "10sec_max_non_overlap": maximum 10-second nonoverlapping epoch
          per minute
        * "30sec_max_non_overlap": maximum 30-second nonoverlapping epoch
          per minute

        Default is "30sec_max_non_overlap".
    threshold: float, optional
        Threshold value for scoring sleep/wake. Default is 1.0.
    rescoring: bool, optional
        If set to True, Webster's rescoring rules are applied [2]_.
        Default is True.

    Returns
    -------
    ck: pandas.core.Series
        Time series containing the `D` scores (0: sleep, 1: wake) for each
        epoch.

    References
    ----------

    .. [1] Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J.,
           & Gillin, J. C. (1992). Automatic Sleep/Wake Identification
           From Wrist Activity. Sleep, 15(5), 461–469.
           http://doi.org/10.1093/sleep/15.5.461
    .. [2] Webster, J. B., Kripke, D. F., Messin, S., Mullaney, D. J., &
           Wyborney, G. (1982). An Activity-Based Sleep Monitor System for
           Ambulatory Use. Sleep, 5(4), 389–399.
           https://doi.org/10.1093/sleep/5.4.389
    """
    available_settings = [
        "mean",
        "10sec_max_overlap",
        "10sec_max_non_overlap",
        "30sec_max_non_overlap"
    ]

    def _cole_kripke(data, scale, window, threshold):
        """Automatic scoring methods from Cole and Kripke"""

        ck = data.rolling(
            window.size, center=True
        ).apply(_window_convolution, args=(scale, window), raw=True)

        return (ck < threshold).astype(int)

    if settings is None:
        settings = "30sec_max_non_overlap"

    if settings not in available_settings:
        raise ValueError("CK sleep/wake identification:\n"
                         + "Required settings: {}\n".format(settings)
                         + "Available settings are:\n"
                         + "\n".join(available_settings))

    ck = None

    if settings == "mean":
        if pd.Timedelta(data.index.freq) > pd.Timedelta('60s'):
            raise ValueError((
                "The sampling frequency of the input data does not allow"
                + " the use of the requested settings ({}).\n".format(
                    settings)
                + "For such settings, the sampling frequency should less"
                + " than 60 seconds."
                ))
        else:
            # Compute the resampling factor as the original weights are
            # calculated for an average of 2-sec epochs over 60 seconds:
            rs_f = 30  # 60sec/2sec
            # Resample to 60 sec and take the sum
            # WARNING: valid if the input data is a sum of activity counts
            data_mean = data.resample('60s').sum()/rs_f

            # Define the scale and weights for this settings
            scale = 0.001
            window = np.array(
                [106, 54, 58, 76, 230, 74, 67, 0, 0],
                np.int32
            )

            ck = _cole_kripke(data_mean, scale, window, threshold)

    elif settings == "10sec_max_overlap":
        if pd.Timedelta(data.index.freq) > pd.Timedelta('5s'):
            raise ValueError((
                "The sampling frequency of the input data does not allow"
                + " the use of the requested settings ({}).\n".format(
                    settings)
                + "For such settings, the sampling frequency should less"
                + " than 5 seconds."
                ))
        else:
            # Resample to 10/2 sec and then sum over a sliding window
            data = data.resample('5s').sum().rolling('10s').sum()
            # Resample to 60 sec and take the max of the 10 sec periods
            data_max = data.resample('60s').max()

            # Define the scale and weights for this settings
            scale = 0.00001
            window = np.array(
                [50, 30, 300, 400, 1400, 500, 350, 0, 0],
                np.int32
            )

            ck = _cole_kripke(data_max, scale, window, threshold)

    elif settings == "10sec_max_non_overlap":
        if pd.Timedelta(data.index.freq) > pd.Timedelta('10s'):
            raise ValueError((
                "The sampling frequency of the input data does not allow"
                + " the use of the requested settings ({}).\n".format(
                    settings)
                + "For such settings, the sampling frequency should less"
                + " or equal to 10 seconds."
                ))
        else:
            # Resample to 10 sec and then sum
            data = data.resample('10s').sum()
            # Resample to 60 sec and take the max of the 10 sec periods
            data_max = data.resample('60s').max()

            # Define the scale and weights for this settings
            scale = 0.00001
            window = np.array(
                [550, 378, 413, 699, 1736, 287, 300, 0, 0],
                np.int32
            )

            ck = _cole_kripke(data_max, scale, window, threshold)

    elif settings == "30sec_max_non_overlap":
        if pd.Timedelta(data.index.freq) > pd.Timedelta('30s'):
            raise ValueError((
                "The sampling frequency of the input data does not allow"
                + " the use of the requested settings ({}).\n".format(
                    settings)
                + "For such settings, the sampling frequency should less"
                + " or equal to 30 seconds."
                ))
        else:
            # Resample to 30 sec and then sum
            data = data.resample('30s').sum()
            # Resample to 60 sec and take the max of the 10 sec periods
            data_max = data.resample('60s').max()

            # Define the scale and weights for this settings
            scale = 0.0001
            window = np.array(
                [50, 30, 14, 28, 12, 8, 50, 0, 0],
                np.int32
            )

            ck = _cole_kripke(data_max, scale, window, threshold)

    if rescoring:
        # Rescoring
        mask = rescore(ck.values, sleep_score=1)
        ck = ck*mask

    return ck


def Sadeh(data, offset=7.601, weights=None, threshold=0.0):
    """Sadeh algorithm for sleep identification

    Algorithm for automatic sleep scoring based on wrist activity,
    developped by Sadeh et al [1]_.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    offset: float, optional
        Offset parameter.
        Default is 7.601.
    weights: np.array
        Array of weighting factors for mean_W5, NAT, sd_Last6 and logAct.
        Default is [-0.065, -1.08, -0.056, -0.703].
    threshold: float, optional
        Threshold value for scoring sleep/wake.
        Default is 0.0.

    Returns
    -------

    References
    ----------

    .. [1] Sadeh, A., Alster, J., Urbach, D., & Lavie, P. (1989).
           Actigraphically based automatic bedtime sleep-wake scoring:
           validity and clinical applications.
           Journal of ambulatory monitoring, 2(3), 209-216.
    .. [2] Sadeh, A., Sharkey, M., & Carskadon, M. A. (1994).
           Activity-Based Sleep-Wake Identification: An Empirical Test of
           Methodological Issues. Sleep, 17(3), 201–207.
           http://doi.org/10.1093/sleep/17.3.201
    """
    if weights is None:
        weights = np.array([-0.065, -1.08, -0.056, -0.703])
    r = data.rolling(11, center=True)

    mean_W5 = r.mean()

    NAT = r.apply(lambda x: np.size(np.where((x > 50) & (x < 100))), raw=True)

    sd_Last6 = data.rolling(6).std()

    logAct = data.shift(-1).apply(lambda x: np.log(1+x))

    sadeh = pd.concat(
        [mean_W5, NAT, sd_Last6, logAct],
        axis=1,
        keys=['mean_W5', 'NAT', 'sd_Last6', 'logAct']
    )

    sadeh['PS'] = sadeh.apply(
        _window_convolution, axis=1, args=(1.0, weights, offset), raw=True
    )

    return (sadeh['PS'] > threshold).astype(int)


def Scripps(data, scale=0.204, window=None, threshold=1.0):
    r"""Scripps Clinic algorithm for sleep-wake identification.

    Algorithm for automatic sleep scoring based on wrist activity,
    developed by Kripke et al [1]_.


    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    scale: float, optional
        Scale parameter P
        Default is 0.204.
    window: np.array, optional
        Array of weighting factors :math:`W_{i}`
        Default values are identical to those found in the original
        publication [1]_.
    threshold: float, optional
        Threshold value for scoring sleep/wake.
        Default is 1.0.

    Returns
    -------

    scripps: pandas.core.Series
        Time series containing the `D` scores (0: sleep, 1: wake) for each
        epoch.

    References
    ----------

    .. [1] Kripke, D. F., Hahn, E. K., Grizas, A. P., Wadiak, K. H.,
           Loving, R. T., Poceta, J. S., … Kline, L. E. (2010).
           Wrist actigraphic scoring for sleep laboratory patients:
           algorithm development. Journal of Sleep Research, 19(4),
           612–619. http://doi.org/10.1111/j.1365-2869.2010.00835.x

    """
    if window is None:
        window = np.array([
            0.0064,  # b_{-10}
            0.0074,  # b_{-9}
            0.0112,  # b_{-8}
            0.0112,  # b_{-7}
            0.0118,  # b_{-6}
            0.0118,  # b_{-5}
            0.0128,  # b_{-4}
            0.0188,  # b_{-3}
            0.0280,  # b_{-2}
            0.0664,  # b_{-1}
            0.0300,  # b_{+0}
            0.0112,  # b_{+1}
            0.0100,  # b_{+2}
            0.0000,  # b_{+3}
            0.0000,  # b_{+4}
            0.0000,  # b_{+5}
            0.0000,  # b_{+6}
            0.0000,  # b_{+7}
            0.0000,  # b_{+8}
            0.0000,  # b_{+9}
            0.0000   # b_{+10}
        ])

    scripps = data.rolling(
        window.size, center=True
    ).apply(_window_convolution, args=(scale, window), raw=True)

    return (scripps < threshold).astype(int)


def Oakley(data, threshold=40):
    r"""Oakley's algorithm for sleep/wake scoring.

    Algorithm for automatic sleep/wake scoring based on wrist activity,
    developed by Oakley [1]_.


    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    threshold: float or str, optional
        Threshold value for scoring sleep/wake. Can be set to "automatic"
        (cf. Notes).
        Default is 40.

    Returns
    -------

    oakley: pandas.core.Series
        Time series containing scores (1: sleep, 0: wake) for each
        epoch.

    References
    ----------

    .. [1] Oakley, N.R. Validation with Polysomnography of the Sleepwatch
           Sleep/Wake Scoring Algorithm Used by the Actiwatch Activity
           Monitoring System; Technical Report; Mini-Mitter: Bend, OR, USA,
           1997
    .. [2] Instruction manual, Actiwatch Communication and Sleep Analysis
           Software
           (https://fccid.io/JIAAWR1/Users-Manual/USERS-MANUAL-1-920937)

    """

    # Sampling frequency
    freq = pd.Timedelta(data.index.freq)

    if freq == pd.Timedelta('15s'):
        window = np.array([
            0.04,  # W_{-8}
            0.04,  # W_{-7}
            0.04,  # W_{-6}
            0.04,  # W_{-5}
            0.20,  # W_{-4}
            0.20,  # W_{-3}
            0.20,  # W_{-2}
            0.20,  # W_{-1}
            4.00,  # W_{+0}
            0.20,  # W_{+1}
            0.20,  # W_{+2}
            0.20,  # W_{+3}
            0.20,  # W_{+4}
            0.04,  # W_{+5}
            0.04,  # W_{+6}
            0.04,  # W_{+7}
            0.04   # W_{+8}
        ], float)
    elif freq == pd.Timedelta('30s'):
        window = np.array([
            0.04,  # W_{-4}
            0.04,  # W_{-3}
            0.20,  # W_{-2}
            0.20,  # W_{-1}
            2.00,  # W_{+0}
            0.20,  # W_{+1}
            0.20,  # W_{+2}
            0.04,  # W_{+3}
            0.04   # W_{+4}
        ], float)
    elif freq == pd.Timedelta('60s'):
        window = np.array([
            0.04,  # W_{-2}
            0.20,  # W_{-1}
            1.00,  # W_{+0}
            0.20,  # W_{+1}
            0.04   # W_{+2}
        ], float)
    elif freq == pd.Timedelta('120s'):
        window = np.array([
            0.12,  # W_{-1}
            0.50,  # W_{+0}
            0.12,  # W_{+1}
        ], float)
    else:
        raise ValueError(
            'Oakley\'s algorithm is not defined for data '
            + 'acquired with a sampling frequency of {}. '.format(freq)
            + 'Accepted frequencies are: {}'.format(
                ', '.join(['15sec', '30sec', '60sec', '120sec'])
            )
        )
    if threshold == 'automatic':
        threshold = _actiware_automatic_threshold(data)
    elif not np.isscalar(threshold):
        msg = "`threshold` should be a scalar or 'automatic'."
        raise ValueError(msg)

    scale = 1.
    oakley = data.rolling(
        window.size, center=True
    ).apply(_window_convolution, args=(scale, window), raw=True)

    return (oakley < threshold).astype(int)


def CSM(ZCMn, settings="auto",score_rest=2,score_sleep=1,binarize=False):
    """Condor Sleep Model

    Sleep-wake scoring algorithm developed by Condor Instrument for their
    ActTrust devices.

    This algorithm works in a two-step fashion. First, it classifies all
    epochs as wake or rest, as function of each epoch's score. Second,
    using a more stringent scoring threshold, "rest" epoch are
    re-classified as "sleep". A relabelling mechanism using the labels of
    the surrounding epochs is also applied to consolidate periods of epochs
    labelled as rest or sleep.

    Parameters
    ----------
    ZCMn : pandas.Series, optional
        Input data series with a DatetimeIndex that contains data from the ZCMn mode in ATR devices.
    settings: str, optional
        Parameter settings for the CSM algorithm. Refers to the data
        acquisition frequency. Available values are:
        * "auto": use input data frequency.
        * "30s": set parameters to optimal values obtained for a 30s data
          acquisition frequency.
        * "60s": set parameters to optimal values obtained for a 60s data
          acquisition frequency.
         Default is 'auto'.
    score_rest: int, optional
        State index for epochs labelled as "rest".
        Default is 2.
    score_sleep: int, optional
        State index for epochs labelled as "sleep".
        Default is 1.
    binarize: bool, optional.
        If set to True, the state index is set to 1 if the epoch is
        labelled as sleep and 0 otherwise.
        Defautl is False.

    Returns
    -------
    csm : pandas.Series
        Series of state indices.
    """
    # This algorithm has been developed for ActTrust devices from Condor Instrument.
    # The CSM uses the ZCMn as input
    data = ZCMn

    if settings == "auto":
        freq = data.index.freq.delta
    else:
        freq = pd.Timedelta(settings)

    if freq == pd.Timedelta('60s'):
        wa = np.array(
            [34.5, 133, 529, 375, 408, 400.5, 1074, 2048.5, 2424.5]
        )
        wp = np.array(
            [1920, 149.5, 257.5, 125, 111.5, 120, 69, 40.5]
        )
    elif freq == pd.Timedelta('30s'):
        wa = np.array(
            [69, 197, 730, 328, 269, 481, 528, 288, 304, 497, 1105, 1043,
             1378, 2719, 2852, 1997]
        )
        wp = np.array(
            [2972, 868, 269, 30, 495, 20, 39, 211, 91, 132, 203, 37, 67,
             71, 81]
        )
    else:
        raise NotImplementedError(
            "The settings for this acquistion frequency ({}) ".format(
                freq
            ) + "have not been implemented yet."
        )

    # The overall scaling and rescoring rules are identical for the
    # different settings.
    p_rest = 0.00005
    p_sleep = 0.000464
    pr_rest = 0
    nr_rest = 0
    pr_sleep = 1
    nr_sleep = 0

    states = csm(
        data,
        wa=wa,
        wp=wp,
        p_rest=p_rest,
        p_sleep=p_sleep,
        pr_rest=pr_rest,
        nr_rest=nr_rest,
        pr_sleep=pr_sleep,
        nr_sleep=nr_sleep,
        score_rest=score_rest,
        score_sleep=score_sleep
    )

    return (states == score_sleep).astype(int) if binarize else states


def SoD(data, whs=4, start='12:00:00', period='5h', algo='Roenneberg', *args, **kwargs):
    r"""Sleep over Daytime

    Quantify the volume of epochs identified as sleep over daytime (SoD),
    using sleep-wake scoring algorithms.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    whs: int, optional
        Window half size. Only valid if start='AonT' or 'AoffT'.
        Default is 4
    start: str, optional
        Start time of the period of interest.
        Default: '12:00:00'
        Supported times: 'AonT', 'AoffT', any 'HH:MM:SS'
    period: str, optional
        Period length.
        Default is '5h'
    algo: str, optional
        Sleep scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    sod: pandas.core.Series
        Time series containing the epochs of rest (1) and
        activity (0) over the specified period.
    """

    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo]

    # Score activity
    sleep_scoring = sleep_algo(data=data, *args, **kwargs)

    # Regex pattern for HH:MM:SS time string
    pattern = re.compile(r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$")

    if start == 'AonT':
        td = AonT(data=data, whs=whs)
    elif start == 'AoffT':
        td = AoffT(data=data, whs=whs)
    elif pattern.match(start):
        td = pd.Timedelta(start)
    else:
        print('Input string for start ({}) not supported.'.format(start))
        return None

    start_time = _td_format(td)
    end_time = _td_format(td+pd.Timedelta(period))

    sod = sleep_scoring.between_time(start_time, end_time)

    return sod


def fSoD(data, whs=12, start='12:00:00', period='5h', algo='Roenneberg', *args, **kwargs):
    r"""Fraction of Sleep over Daytime

    Fractional volume of epochs identified as sleep over daytime (SoD),
    using sleep-wake scoring algorithms.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    whs: int, optional
        Window half size.
        Default is 4
    start: str, optional
        Start time of the period of interest.
        Supported times: 'AonT', 'AoffT', any 'HH:MM:SS'.
        Default: '12:00:00'.
    period: str, optional
        Period length.
        Default is '10h'.
    algo: str, optional
        Sleep scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    fsod: float
        Fraction of epochs scored as sleep, relatively to the length of
        the specified period.
    """
    sod = SoD(data=data, whs=whs,start=start, period=period,algo=algo, *args, **kwargs)
    return sod.sum()/len(sod)


def Crespo(
    data,
    frequency,
    zeta=15,
    zeta_r=30,
    zeta_a=2,
    t=.33,
    alpha='8h',
    beta='1h',
    estimate_zeta=False,
    seq_length_max=100,
    verbose=False
):
    r"""Crespo algorithm for activity/rest identification

    Algorithm for automatic identification of activity-rest periods based
    on actigraphy, developed by Crespo et al. [1]_.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    frequency: int
        Sampling frequency of the actigraphy trace.
    zeta: int, optional
        Maximum number of consecutive zeroes considered valid.
        Default is 15.
    zeta_r: int, optional
        Maximum number of consecutive zeroes considered valid (rest).
        Default is 30.
    zeta_a: int, optional
        Maximum number of consecutive zeroes considered valid (active).
        Default is 2.
    t: float, optional
        Percentile for invalid zeroes.
        Default is 0.33.
    alpha: str, optional
        Average hours of sleep per night.
        Default is '8h'.
    beta: str, optional
        Length of the padding sequence used during the processing.
        Default is '1h'.
    estimate_zeta: bool, optional
        If set to True, zeta values are estimated from the distribution of
        ratios of the number of series of consecutive zeroes to
        the number of series randomly chosen from the actigraphy data.
        Default is False.
    seq_length_max: int, optional
        Maximal length of the aforementioned random series.
        Default is 100.
    verbose: bool, optional
        If set to True, print the estimated values of zeta.
        Default is False.

    Returns
    -------
    crespo : pandas.core.Series
        Time series containing the estimated periods of rest (0) and
        activity (1).

    References
    ----------

    .. [1] Crespo, C., Aboy, M., Fernández, J. R., & Mojón, A. (2012).
           Automatic identification of activity–rest periods based on
           actigraphy. Medical & Biological Engineering & Computing, 50(4),
           329–340. http://doi.org/10.1007/s11517-012-0875-y
    """

    # 1. Pre-processing
    # This stage produces an initial estimate of the rest-activity periods

    # 1.1. Signal conditioning based on empirical probability model
    # This step replaces sequences of more than $\zeta$ "zeroes"
    # with the t-percentile value of the actigraphy data
    # zeta = 15
    if estimate_zeta:
        zeta = _estimate_zeta(data, seq_length_max)
        if verbose:
            print("CRESPO: estimated zeta = {}".format(zeta))
    # Determine the sequences of consecutive zeroes
    mask_zeta = _create_inactivity_mask(data, zeta, 1)

    # Determine the user-specified t-percentile value
    s_t = data.quantile(t)

    # Replace zeroes with the t-percentile value
    x = data.copy()
    x[mask_zeta > 0] = s_t

    # Median filter window length L_w
    L_w = int(pd.Timedelta(alpha)/frequency)+1
    L_w_over_2 = int((L_w-1)/2)

    # Pad the signal at the beginning and at the end with a sequence of
    # $\alpha/2$ h of elements of value $m = max(s(t))$.
    #
    # alpha_epochs = int(pd.Timedelta(alpha)/self.frequency)
    # alpha_epochs_half = int(alpha_epochs/2)
    # beta_epochs = int(pd.Timedelta(beta)/self.frequency)

    s_t_max = data.max()

    x_p = _padded_data(data, s_t_max, L_w_over_2, frequency)

    # 1.2 Rank-order processing and decision logic
    # Apply a median filter to the $x_p$ series
    x_f = x_p.rolling(L_w, center=True, min_periods=L_w_over_2).median()

    # Rank-order thresholding
    # Create a series $y_1(n)$ where $y_1(n) = 1$ for $x_f(n)>p$, $0$ otw.
    # The threshold $p$ is the percentile of $x_f(n)$ corresponding to
    # $(h_s/24)\times 100\%$

    p_threshold = x_f.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))
    y_1 = pd.Series(np.where(x_f > p_threshold, 1, 0), index=x_f.index)

    # 1.3 Morphological filtering

    # Morph. filter window length, L_p
    L_p = int(pd.Timedelta(beta)/frequency)+1

    # Morph. filter, M_f
    M_f = np.ones(L_p)

    # Apply a morphological closing operation

    y_1_close = binary_closing(y_1, M_f).astype(int)

    # Apply a morphological opening operation

    y_1_close_and_open = binary_opening(y_1_close, M_f).astype(int)

    y_e = pd.Series(y_1_close_and_open, index=y_1.index)

    # 2. Processing and decision logic
    # This stage uses the estimates of the rest-activity periods
    # from the previous stage.

    # 2.1 Model-based data validation

    # Create a mask for sequences of more than $\zeta_{rest}$ zeros
    # during the rest periods

    # zeta_r = 30
    # zeta_a = 2
    if estimate_zeta:
        zeta_r = _estimate_zeta(data[y_e < 1], seq_length_max)
        zeta_a = _estimate_zeta(data[y_e > 0], seq_length_max)
        if verbose:
            print("CRESPO: estimated zeta@rest= {}".format(zeta_r))
            print("CRESPO: estimated zeta@actv= {}".format(zeta_a))

    # Find sequences of zeroes during the rest and the active periods
    # and mark as invalid sequences of more $\zeta_x$ zeroes.

    # Use y_e series as a filter for the rest periods
    mask_rest = _create_inactivity_mask(data[y_e < 1], zeta_r, 1)

    # Use y_e series as a filter for the active periods
    mask_actv = _create_inactivity_mask(data[y_e > 0], zeta_a, 1)

    mask = pd.concat([mask_actv, mask_rest], verify_integrity=True)

    # 2.2 Adaptative rank-order processing and decision logic

    # Replace masked values by NaN so that they are not taken into account
    # by the median filter.
    # Done before padding to avoid unaligned time series.

    x_nan = data.copy()
    x_nan[mask < 1] = np.NaN

    # Pad the signal at the beginning and at the end with a sequence of 1h
    # of elements of value m = max(s(t)).
    x_sp = _padded_data(x_nan, s_t_max, L_p-1, frequency)

    # Apply an adaptative median filter to the $x_{sp}$ series
    # no need to use a time-aware window as there is no time gap
    # in this time series by definition.
    x_fa = x_sp.rolling(L_w, center=True, min_periods=L_p-1).median()

    # The 'alpha' hour window is biased at the edges as it is not
    # symmetrical anymore. In the regions (start, start+alpha/2,
    # the median needs to be calculate by hand.
    # The range is start, start+alpha as the window is centered.
    median_start = x_sp.iloc[0:L_w].expanding().median()
    median_end = x_sp.iloc[-L_w-1:-1].sort_index(ascending=False).expanding().median()[::-1]

    # replace values in the original x_fa series with the new values
    # within the range (start, start+alpha/2) only.
    x_fa.iloc[0:L_w_over_2] = median_start.iloc[0:L_w_over_2]
    x_fa.iloc[-L_w_over_2-1:-1] = median_end.iloc[0:L_w_over_2]

    # restore original time range
    x_fa = x_fa[data.index[0]:data.index[-1]]

    p_threshold = x_fa.quantile((pd.Timedelta(alpha)/pd.Timedelta('24h')))

    y_2 = pd.Series(np.where(x_fa > p_threshold, 1, 0), index=x_fa.index)

    # ### 2.3 Morphological filtering
    y_2_close = binary_closing(y_2, structure=np.ones(2*(L_p-1)+1)).astype(int)

    y_2_open = binary_opening(y_2_close, structure=np.ones(2*(L_p-1)+1)).astype(int)

    crespo = pd.Series(y_2_open, index=y_2.index)

    # Manual post-processing
    crespo.iloc[0] = 1
    crespo.iloc[-1] = 1

    return crespo

def Crespo_AoT(
    data,
    frequency,
    zeta=15,
    zeta_r=30,
    zeta_a=2,
    t=.33,
    alpha='8h',
    beta='1h',
    estimate_zeta=False,
    seq_length_max=100,
    verbose=False
):
    """Automatic identification of activity onset/offset times, based on
    the Crespo algorithm.

    Identification of the activity onset and offset times using the
    algorithm for automatic identification of activity-rest periods based
    on actigraphy, developed by Crespo et al. [1]_.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    frequency: int
        Sampling frequency of the actigraphy trace.
    zeta: int
        Maximum number of consecutive zeroes considered valid.
        Default is 15.
    zeta_r: int
        Maximum number of consecutive zeroes considered valid (rest).
        Default is 30.
    zeta_a: int
        Maximum number of consecutive zeroes considered valid (active).
        Default is 2.
    t: float
        Percentile for invalid zeroes.
        Default is 0.33.
    alpha: offset
        Average hours of sleep per night.
        Default is '8h'.
    beta: offset
        Length of the padding sequence used during the processing.
        Default is '1h'.
    estimate_zeta: Boolean
        If set to True, zeta values are estimated from the distribution of
        ratios of the number of series of consecutive zeroes to
        the number of series randomly chosen from the actigraphy data.
        Default is False.
    seq_length_max: int
        Maximal length of the aforementioned random series.
        Default is 100.
    verbose:
        If set to True, print the estimated values of zeta.
        Default is False.

    Returns
    -------
    aot : (ndarray, ndarray)
        Arrays containing the estimated activity onset and offset times,
        respectively.

    References
    ----------

    .. [1] Crespo, C., Aboy, M., Fernández, J. R., & Mojón, A. (2012).
           Automatic identification of activity–rest periods based on
           actigraphy. Medical & Biological Engineering & Computing, 50(4),
           329–340. http://doi.org/10.1007/s11517-012-0875-y

    """
    crespo = Crespo(
        data=data,
        frequency=frequency,
        zeta=zeta,
        zeta_r=zeta_r,
        zeta_a=zeta_a,
        t=t,
        alpha=alpha,
        beta=beta,
        estimate_zeta=estimate_zeta,
        seq_length_max=seq_length_max,
        verbose=verbose
    )

    diff = crespo.diff(1)

    AonT = crespo[diff == 1].index
    AoffT = crespo[diff == -1].index

    return (AonT, AoffT)

def Roenneberg(
    data,
    trend_period='24h',
    min_trend_period='12h',
    threshold=0.15,
    min_seed_period='30Min',
    max_test_period='12h',
    r_consec_below='30Min'
):
    """Automatic sleep detection.

    Identification of consolidated sleep episodes using the
    algorithm developed by Roenneberg et al. [1].

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    trend_period: str, optional
        Time period of the rolling window used to extract the data trend.
        Default is '24h'.
    min_trend_period: str, optional
        Minimum time period required for the rolling window to produce a
        value. Values default to NaN otherwise.
        Default is '12h'.
    threshold: float, optional
        Fraction of the trend to use as a threshold for sleep/wake
        categorization.
        Default is '0.15'
    min_seed_period: str, optional
        Minimum time period required to identify a potential sleep onset.
        Default is '30Min'.
    max_test_period : str, optional
        Maximal period of the test series.
        Default is '12h'
    r_consec_below : str, optional
        Time range to consider, past the potential correlation peak when
        searching for the maximum correlation peak.
        Default is '30Min'.

    Returns
    -------
    rbg : pandas.core.Series
        Time series containing the estimated periods of rest (1) and
        activity (0).

    Notes
    -----

    .. warning:: The performance of this algorithm has been evaluated for
                 actigraphy data aggregated in 10-min bins [2]_.

    References
    ----------

    .. [1] Roenneberg, T., Keller, L. K., Fischer, D., Matera, J. L.,
           Vetter, C., & Winnebeck, E. C. (2015). Human Activity and Rest
           In Situ. In Methods in Enzymology (Vol. 552, pp. 257-283).
           http://doi.org/10.1016/bs.mie.2014.11.028
    .. [2] Loock, A., Khan Sullivan, A., Reis, C., Paiva, T., Ghotbi, N.,
           Pilz, L. K., Biller, A. M., Molenda, C., Vuori‐Brodowski, M. T.,
           Roenneberg, T., & Winnebeck, E. C. (2021). Validation of the
           Munich Actimetry Sleep Detection Algorithm for estimating
           sleep–wake patterns from activity recordings. Journal of Sleep
           Research, April, 1–12. https://doi.org/10.1111/jsr.13371
    """
    rbg = roenneberg(
        data,
        trend_period=trend_period,
        min_trend_period=min_trend_period,
        threshold=threshold,
        min_seed_period=min_seed_period,
        max_test_period=max_test_period,
        r_consec_below=r_consec_below
    )

    return rbg

def Roenneberg_AoT(
    data,
    trend_period='24h',
    min_trend_period='12h',
    threshold=0.15,
    min_seed_period='30Min',
    max_test_period='12h',
    r_consec_below='30Min'
):
    """Automatic identification of activity onset/offset times, based on
    Roenneberg's algorithm.

    Identification of the activity onset and offset times using the
    algorithm for automatic identification of consolidated sleep episodes
    developped by Roenneberg et al. [1]_.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    trend_period: str, optional
        Time period of the rolling window used to extract the data trend.
        Default is '24h'.
    min_trend_period: str, optional
        Minimum time period required for the rolling window to produce a
        value. Values default to NaN otherwise.
        Default is '12h'.
    threshold: float, optional
        Fraction of the trend to use as a threshold for sleep/wake
        categorization.
        Default is '0.15'
    min_seed_period: str, optional
        Minimum time period required to identify a potential sleep onset.
        Default is '30Min'.
    max_test_period : str, optional
        Maximal period of the test series.
        Default is '12h'
    r_consec_below : str, optional
        Time range to consider, past the potential correlation peak when
        searching for the maximum correlation peak.
        Default is '30Min'.

    Returns
    -------
    aot : (ndarray, ndarray)
        Arrays containing the estimated activity onset and offset times,
        respectively.

    Notes
    -----

    .. warning:: The performance of this algorithm has been evaluated for
                 actigraphy data aggregated in 10-min bins [2]_.

    References
    ----------

    .. [1] Roenneberg, T., Keller, L. K., Fischer, D., Matera, J. L.,
           Vetter, C., & Winnebeck, E. C. (2015). Human Activity and Rest
           In Situ. In Methods in Enzymology (Vol. 552, pp. 257-283).
           http://doi.org/10.1016/bs.mie.2014.11.028
    .. [2] Loock, A., Khan Sullivan, A., Reis, C., Paiva, T., Ghotbi, N.,
           Pilz, L. K., Biller, A. M., Molenda, C., Vuori‐Brodowski, M. T.,
           Roenneberg, T., & Winnebeck, E. C. (2021). Validation of the
           Munich Actimetry Sleep Detection Algorithm for estimating
           sleep–wake patterns from activity recordings. Journal of Sleep
           Research, April, 1–12. https://doi.org/10.1111/jsr.13371

    Examples
    --------

    """

    rbg = Roenneberg(
        data=data,
        trend_period=trend_period,
        min_trend_period=min_trend_period,
        threshold=threshold,
        min_seed_period=min_seed_period,
        max_test_period=max_test_period,
        r_consec_below=r_consec_below
    )

    diff = rbg.diff(1)

    AonT = rbg[diff == -1].index
    AoffT = rbg[diff == 1].index

    return (AonT, AoffT)


def SleepProfile(data, freq='15min', algo='Roenneberg', *args, **kwargs):
    r"""Normalized sleep daily profile


    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    freq: str, optional
        Resampling frequency.
        Default is '15min'
    algo: str, optional
        Sleep scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    sleep_prof: YYY

    """

    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo]

    # Score activity
    sleep_scoring = sleep_algo(data=data, *args, **kwargs)

    # Sleep profile over 24h
    sleep_prof = _average_daily_activity(data=sleep_scoring, cyclic=False)

    # Retrun resampled sleep profile
    return sleep_prof.resample(freq).mean()


def SleepRegularityIndex(data, bin_threshold=None, algo='Roenneberg', *args, **kwargs):
    r""" Sleep regularity index

    Likelihood that any two time-points (epoch-by-epoch) 24 hours apart are
    in the same sleep/wake state, across all days. This index is originally
    defined in [1]_ and validated in older adults in [2]_.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    freq: str, optional
        Resampling frequency.
        Default is '15min'
    bin_threshold: bool, optional
        If bin_threshold is not set to None, scoring data above this
        threshold are set to 1 and to 0 otherwise.
        Default is None.
    algo: str, optional
        Sleep scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    sri: float

    Notes
    -----

    The sleep regularity index (SRI) is defined as:

    .. math::

        SRI = -100 + \frac{200}{M(N-1)} \sum_{j=1}^M\sum_{i=1}^N
              \delta(s_{i,j}, s_{i+1,j})

    with:
        :math:`\delta(s_{i,j}, s_{i+1,j}) = 1` if
        :math:`s_{i,j} = s_{i+1,j}` and 0 otherwise.

    References
    ----------

    .. [1] Phillips, A. J. K., Clerx, W. M., O’Brien, C. S., Sano, A.,
           Barger, L. K., Picard, R. W., … Czeisler, C. A. (2017).
           Irregular sleep/wake patterns are associated with poorer
           academic performance and delayed circadian and sleep/wake
           timing. Scientific Reports, 7(1), 1–13.
           https://doi.org/10.1038/s41598-017-03171-4
    .. [2] Lunsford-Avery, J. R., Engelhard, M. M., Navar, A. M.,
           & Kollins, S. H. (2018). Validation of the Sleep Regularity
           Index in Older Adults and Associations with Cardiometabolic
           Risk. Scientific Reports, 8(1), 14158.
           https://doi.org/10.1038/s41598-018-32402-5
    """
    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo]

    # Score activity
    sleep_scoring = sleep_algo(data=data, *args, **kwargs)

    # Return sleep Regularity Index (SRI)
    return sri(sleep_scoring, bin_threshold)


def SleepMidPoint(data, bin_threshold=None, to_td=True, algo='Roenneberg', *args, **kwargs):
    r""" Sleep midpoint

    Center of the mean sleep periods

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    bin_threshold: bool, optional
        If bin_threshold is not set to None, scoring data above this
        threshold are set to 1 and to 0 otherwise.
        Default is None.
    to_td: bool, optional
        If set to true, the sleep midpoint is returned as a Timedelta.
        Otherwise, it represents the number of minutes since midnight.
    algo: str, optional
        Sleep scoring algorithm to use.
        Default is 'Roenneberg'.
    \*args
        Variable length argument list passed to the scoring algorithm.
    \*\*kwargs
        Arbitrary keyword arguments passed to the scoring algorithm.

    Returns
    -------
    smp: float or Timedelta


    References
    ----------

    .. [1] Lunsford-Avery, J. R., Engelhard, M. M., Navar, A. M.,
           & Kollins, S. H. (2018). Validation of the Sleep Regularity
           Index in Older Adults and Associations with Cardiometabolic
           Risk. Scientific Reports, 8(1), 14158.
           https://doi.org/10.1038/s41598-018-32402-5
    """

    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo]

    # Score activity
    sleep_scoring = sleep_algo(data=data, *args, **kwargs)

    # Sleep midpoint
    smp = sleep_midpoint(sleep_scoring, bin_threshold)

    return pd.Timedelta(smp, unit='min') if to_td is True else smp


def sleep_bouts(data, duration_min=None, duration_max=None, algo='Roenneberg', *args, **kwargs):
    r"""Sleep bouts.

    Activity periods identified as sleep.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    duration_min: str,optional
        Minimal time duration for a sleep period.
        Default is None (no filtering).
    duration_max: str,optional
        Maximal time duration for a sleep period.
        Default is None (no filtering).
    algo: str, optional
        Sleep/wake scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    sleep_bouts: a list of pandas.Series
    """
    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo+'_AoT']

    # Detect activity onset and offset times
    onsets, offsets = sleep_algo(data=data, *args, **kwargs)

    # For each inactivity period (from offset to onset times)
    sleep_bouts = []
    for onset, offset in zip(onsets, offsets):
        sleep_bout = data[offset:onset]
        sleep_bouts.append(sleep_bout)

    return filter_ts_duration(sleep_bouts, duration_min, duration_max)


def active_bouts(data, duration_min=None, duration_max=None, algo='Roenneberg', *args, **kwargs):
    r"""Active bouts.

    Activity periods identified as active.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    duration_min: str,optional
        Minimal time duration for an active period.
        Default is None (no filtering).
    duration_max: str,optional
        Maximal time duration for an active period.
        Default is None (no filtering).
    algo: str, optional
        Sleep/wake scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    active_bouts: a list of pandas.Series
    """
    # Retrieve sleep scoring function dynamically by name
    sleep_algo = globals()[algo+'_AoT']

    # Detect activity onset and offset times
    onsets, offsets = sleep_algo(data=data, *args, **kwargs)

    # Check if first onset occurs after the first offset
    assert offsets[0] < onsets[0]

    # For each activity period (from onset to offset times)
    #  - Deal with first and last active periods manually

    active_bouts = []

    # First active bout (from the beginning of recording to first offset)
    active_bouts.append(data[:offsets[0]])

    for onset, offset in zip(onsets[:-1], offsets[1:]):
        active_bout = data[onset:offset]
        active_bouts.append(active_bout)

    # Last active bout (from last onset to the end of the recording)
    active_bouts.append(data[onsets[-1]:])

    return filter_ts_duration(active_bouts, duration_min, duration_max)


def sleep_durations(data, duration_min=None, duration_max=None, algo='Roenneberg', *args, **kwargs):
    r"""Duration of the sleep bouts.

    Duration of the activity periods identified as sleep.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    duration_min: str,optional
        Minimal time duration for a sleep period.
        Default is None (no filtering).
    duration_max: str,optional
        Maximal time duration for a sleep period.
        Default is None (no filtering).
    algo: str, optional
        Sleep/wake scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    sleep_durations: a list of pandas.TimeDelta
    """

    # Retrieve sleep bouts
    filtered_bouts = sleep_bouts(
        data=data,
        duration_min=duration_min,
        duration_max=duration_max,
        algo=algo,
        *args,
        **kwargs
    )

    return [s.index[-1]-s.index[0] for s in filtered_bouts]


def active_durations(data, duration_min=None, duration_max=None, algo='Roenneberg', *args, **kwargs):
    r"""Duration of the active bouts.

    Duration of the activity periods identified as active.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    duration_min: str,optional
        Minimal time duration for an active period.
        Default is None (no filtering).
    duration_max: str,optional
        Maximal time duration for an active period.
        Default is None (no filtering).
    algo: str, optional
        Sleep/wake scoring algorithm to use.
        Default is 'Roenneberg'.
    *args
        Variable length argument list passed to the scoring algorithm.
    **kwargs
        Arbitrary keyword arguements passed to the scoring algorithm.

    Returns
    -------
    active_durations: a list of pandas.TimeDelta
    """
    # Retrieve sleep bouts
    filtered_bouts = active_bouts(
        data=data,
        duration_min=duration_min,
        duration_max=duration_max,
        algo=algo,
        *args,
        **kwargs
    )

    return [s.index[-1]-s.index[0] for s in filtered_bouts]


def main_sleep_bouts(data, report='major'):
    """
    Calculate main sleep episodes using the Roenneberg algorithm.

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    report : str, optional
        Either 'major' or 'minor'. Default is 'major'. If set to 'major', the function will
        return a dataframe containing all the major sleep bouts in the recording, along
        with the mean. If set to 'minor', the function will return a dataframe containing
        all the minor sleep bouts in the recording, along with the mean.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the main sleep episodes (date, start_time, stop_time and duration).

    """
    # Compute the activity onset and off using the Roenneberg algorithm
    activity_onset, activity_offset = Roenneberg_AoT(data)

    # Create empty dataframe to store all the sleep events
    sleep_events = pd.DataFrame()

    # sleep_onset = activity_offset; sleep_offset = activity_onset
    sleep_events['date'] = activity_onset.date
    sleep_events['start_time'] = activity_offset
    sleep_events['stop_time'] = activity_onset

    # Sleep/rest episode duration
    sleep_events['duration'] = sleep_events['stop_time'] - sleep_events['start_time']

    # Identify main sleep episode
    main_sleep = sleep_events.loc[sleep_events.groupby('date')['duration'].idxmax()]

    # Identify minor sleep episodes
    minor_sleep = sleep_events.drop(main_sleep.index)

    if report == 'major':
        # Calculate mean duration of the main sleep episode
        #mean = main_sleep['duration'].mean().total_seconds() / 60
        mean = np.mean(main_sleep['duration'])

        # Return dataframe with major sleep events and summary stats
        return main_sleep, mean

    elif report == 'minor':
        # Calculate mean duration of the main sleep episode
        #mean = minor_sleep['duration'].mean().total_seconds() / 60
        mean = np.mean(minor_sleep['duration'])

        # Return dataframe with major sleep events and summary stats
        return minor_sleep, mean

def waso(data, algo='Cole-Kripke', **kwargs):
    """
    Calculate Wake After Sleep Onset (WASO)

    Parameters
    ----------
    data : pandas.Series, optional
        Input data series with a DatetimeIndex, where the index specifies the time points and
        the values represent the input variable (e.g., activity, light). Time and value arrays
        are extracted from this series.
    algo
        Sleep detection algorithm to use to detect sleep fragments during a consolidated sleep
        period (as determined by the Roenneberg algorithm). It can be either 'Cole-Kripke',
        'Sadeh' or 'Scripps'.
    **kwargs
        Pass the 'settings' keyword argument to if Cole-Kripke is used. Available settings are:
        * "mean": mean activity per minute
        * "10sec_max_overlap": maximum 10-second overlapping epoch per minute
        * "10sec_max_non_overlap": maximum 10-second non-overlapping epoch per minute
        * "30sec_max_non_overlap": maximum 30-second non-overlapping epoch per minute


    Returns
    -------
    pd.Series
        A series containing WASO values per day
    np.float64
        Mean WASO value


    """
    # Calculate main consolidated sleep episodes using the Roenneberg algorithm
    main_sleep_df = main_sleep_bouts(data)[0]

    # Initialize the sleep_flags variable to store sleep segments
    sleep_flags = None

    # Select an algorithm to detect sleep segments
    match algo:
        case 'Cole-Kripke':
            # User must provide a 'settings' argument to use Cole-Kripke
            if 'settings' not in kwargs.keys():
                # In case no setting is provided, use the 'mean' and warn
                warnings.warn('No settings provided, using "mean" instead')
                kwargs['settings'] = 'mean'
            # Populate sleep_flags with the results from 'Cole-Kripke'
            sleep_flags = Cole_Kripke(data, settings=kwargs.get('settings', {}))
        case 'Sadeh':
            # Populate sleep_flags with the results from 'Sadeh'
            sleep_flags = Sadeh(data)
        case 'Scripps':
            # Populate sleep_flags with the results from 'Scripps'
            sleep_flags = Scripps(data)
        case _:
            # Raise an error in case the algorithm selected is not available
            raise ValueError('Algorithm {} is not valid.'.format(algo))

    # Create an empty list to store daily waso values
    waso_values = {}

    # Iterate over the main sleep episodes in the recording
    for _, row in main_sleep_df.iterrows():
        # Extract the date from the current row
        date = row['START'].date()

        # Use the consolidated sleep episode to define the start and stop borders
        sleep_window = sleep_flags[row['start_time']:row['stop_time']]

        # Count minutes in which individual is awake during the consolidated sleep window
        # (0: sleep, 1: wake)
        waso_minutes = ((1 - sleep_window).sum())

        # Append the result to the waso_values list
        waso_values[date] = waso_minutes

    waso_values = pd.Series(waso_values)
    return waso_values, np.mean(waso_values)
