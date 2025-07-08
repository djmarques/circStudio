import pandas as pd
import numpy as np
import re

from ..analysis.tools import *
from ..analysis.metrics import *

from statistics import mean
import statsmodels.api as sm


class Activity(object):
    """Class containing all the functions to compute activity metrics."""

    def average_daily_activity(
        self, freq="5min", cyclic=False, time_origin=None, whs="1h"
    ):
        r"""Average daily activity distribution

        Calculate the daily profile of activity. Data are averaged over all the
        days.

        Parameters
        ----------
        freq: str, optional
            Data resampling frequency.
            Cf. #timeseries-offset-aliases in
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>.
        cyclic: bool, optional
            If set to True, two daily profiles are concatenated to ensure
            continuity between the last point of the day and the first one.
            Default is False.
        time_origin: str or pd.Timedelta, optional
            If not None, origin of the time axis for the daily profile.
            Original time bins are translated as time delta with respect to
            this new origin.
            Default is None
            Supported time string: 'AonT', 'AoffT', any 'HH:MM:SS'
        whs: str, optional
            Window half size parameter for the detection of the activity
            onset/offset time. Relevant only if time_origin is set to
            'AonT' or AoffT'.
            Default is '1h'.

        Returns
        -------
        raw : pandas.Series
            A Series containing the daily activity profile with a 24h index.
        """
        data = self.resample(data=self.activity, freq=freq)
        return daily_profile(
            data=data, freq=freq, cyclic=cyclic, time_origin=time_origin, whs=whs
        )

    def ADAT(self, rescale=True, exclude_ends=False):
        """Total average daily activity

        Calculate the total activity counts, averaged over all the days.

        Parameters
        ----------
        rescale: bool, optional
            If set to True, the activity counts are rescaled to account for
            masked periods (if any).
            Default is True.
        exclude_ends: bool, optional
            If set to True, the first and last daily periods are excluded from
            the calculation. Useful when the recording does start or end at
            midnigth.
            Default is False.

        Returns
        -------
        adat : int
        """
        data = self.resample(data=self.activity, freq=freq)
        return adat(data=data, rescale=rescale, exclude_ends=exclude_ends)

    def adatp(
        self, period="7D", freq=None, rescale=True, exclude_ends=False, verbose=False
    ):
        """Total average daily activity per period

        Calculate the total activity counts, averaged over each consecutive
        period contained in the data. The number of periods

        Parameters
        ----------
        period: str, optional
            Time length of the period to be considered. Must be understandable
            by pandas.Timedelta
        rescale: bool, optional
            If set to True, the activity counts are rescaled to account for
            masked periods (if any).
            Default is True.
        exclude_ends: bool, optional
            If set to True, the first and last daily periods are excluded from
            the calculation. Useful when the recording does start or end at
            midnigth.
            Default is False.
        verbose: bool, optional
            If set to True, display the number of periods found in the data.
            Also display the time not accounted for.
            Default is False.

        Returns
        -------
        adatp : list of int
        """
        data = self.resample(data=self.activity, freq=freq)
        return adatp(
            data=data,
            period=period,
            rescale=rescale,
            exclude_ends=exclude_ends,
            verbose=verbose,
        )

    def l5(self, freq=None):
        r"""L5

        Mean activity during the 5 least active hours of the day.

        Returns
        -------
        l5: float

        Notes
        -----

        The L5 [1]_ variable is calculated as the mean, per acquisition period,
        of the average daily activities during the 5 least active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """

        data = self.resample(data=self.activity, freq=freq)
        return l5(data=data)

    def m10(self, freq=None):
        r"""M10

        Mean activity during the 10 most active hours of the day.

        Returns
        -------
        m10: float

        Notes
        -----

        The M10 [1]_ variable is calculated as the mean, per acquisition period
        , of the average daily activities during the 10 most active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """

        data = self.resample(data=self.activity, freq=freq)
        _, m10 = _lmx(data, "10h", lowest=False)

        return m10

    def relative_amplitude(self, freq=None):
        r"""Relative rest/activity amplitude

        Relative amplitude between the mean activity during the 10 most active
        hours of the day and the mean activity during the 5 least active hours
        of the day.

        Returns
        -------
        ra: float

        Notes
        -----

        The RA [1]_ variable is calculated as:

        .. math::

            RA = \frac{M10 - L5}{M10 + L5}

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """

        data = self.resample(data=self.activity, freq=freq)
        return relative_amplitude(data=data)

    def l5p(self, period="7D", freq=None, verbose=False):
        r"""L5 per period

        The L5 variable is calculated for each consecutive period found in the
        actigraphy recording.

        Parameters
        ----------
        period: str, optional
            Time period for the calculation of IS
            Default is '7D'.
        verbose: bool, optional
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.

        Returns
        -------
        l5p: list of float


        Notes
        -----

        The L5 [1]_ variable is calculated as the mean, per acquisition period,
        of the average daily activities during the 5 least active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """
        data = self.resample(data=self.activity, freq=freq)
        return l5p(data=data, period=period, verbose=verbose)

    def m10p(self, period="7D", freq=None, verbose=False):
        r"""M10 per period

        The M10 variable is calculated for each consecutive period found in the
        actigraphy recording.

        Parameters
        ----------
        period: str, optional
            Time period for the calculation of IS
            Default is '7D'.
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.
        verbose: bool, optional
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.

        Returns
        -------
        m10p: list of float


        Notes
        -----

        The M10 [1]_ variable is calculated as the mean, per acquisition period
        , of the average daily activities during the 10 most active hours.

        .. warning:: The value of this variable depends on the length of the
                     acquisition period.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """
        data = self.resample(data=self.activity, freq=freq)
        intervals = _interval_maker(data.index, period, verbose)
        return m10p(data=data, period=period, verbose=verbose)

    def relative_amplitude_per_period(self, period="7D", freq=None, verbose=False):
        r"""RA per period

        The RA variable is calculated for each consecutive period found in the
        actigraphy recording.

        Parameters
        ----------
        period: str, optional
            Time period for the calculation of IS
            Default is '7D'.
        verbose: bool, optional
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.

        Returns
        -------
        rap: list of float

        Notes
        -----

        The RA [1]_ variable is calculated as:

        .. math::

            RA = \frac{M10 - L5}{M10 + L5}

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206
        """
        data = self.resample(data=self.activity, freq=freq)
        return relative_amplitude_by_period(data=data, period=period, verbose=verbose)

    def interdaily_stability(self, freq="1h"):
        r"""Interdaily stability

        The Interdaily stability (IS) quantifies the repeatibilty of the
        daily rest-activity pattern over each day contained in the activity
        recording.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
            Default is '1h'.

        Returns
        -------
        is: float

        Notes
        -----

        This variable is defined in ref [1]_:

        .. math::

            IS = \frac{d^{24h}}{d^{1h}}

        with:

        .. math::

            d^{1h} = \sum_{i}^{n}\frac{\left(x_{i}-\bar{x}\right)^{2}}{n}

        where :math:`x_{i}` is the number of active (counts higher than a
        predefined threshold) minutes during the :math:`i^{th}` period,
        :math:`\bar{x}` is the mean of all data and :math:`n` is the number of
        periods covered by the actigraphy data and with:

        .. math::

            d^{24h} = \sum_{i}^{p} \frac{
                      \left( \bar{x}_{h,i} - \bar{x} \right)^{2}
                      }{p}

        where :math:`\bar{x}^{h,i}` is the average number of active minutes
        over the :math:`i^{th}` period and :math:`p` is the number of periods
        per day. The average runs over all the days.

        For the record, tt is the 24h value from the chi-square periodogram
        (Sokolove and Bushel1 1978).

        References
        ----------

        .. [1] Witting W., Kwa I.H., Eikelenboom P., Mirmiran M., Swaab D.F.
               Alterations in the circadian rest–activity rhythm in aging and
               Alzheimer׳s disease. Biol Psychiatry. 1990;27:563–572.
        """
        data = self.resample(data=self.activity, freq=freq)
        return interdaily_stability(data=data)

    def average_interdaily_stability(self, freqs=None):
        r"""Average interdaily stability

        ISm [1]_ is the average of the IS values obtained with resampling
        periods divisors of 1440 between 1 and 60 min.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency strings
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is True.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is set to 4.

        Returns
        -------
        ism: float

        Notes
        -----

        By default, the resampling periods are 1, 2, 3, 4, 5, 6, 8, 9, 10, 12,
        15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48 and 60 min.

        References
        ----------

        .. [1] Gonçalves, B. S., Cavalcanti, P. R., Tavares, G. R.,
               Campos, T. F., & Araujo, J. F. (2014). Nonparametric methods in
               actigraphy: An update. Sleep science (Sao Paulo, Brazil), 7(3),
               158-64.
        """
        if freqs is None:
            freqs = [
                "1min",
                "2min",
                "3min",
                "4min",
                "5min",
                "6min",
                "8min",
                "9min",
                "10min",
                "12min",
                "15min",
                "16min",
                "18min",
                "20min",
                "24min",
                "30min",
                "32min",
                "36min",
                "40min",
                "45min",
                "48min",
                "60min",
            ]
        data = [self.resample(data=self.activity, freq=freq) for freq in freqs]
        return mean([interdaily_stability(datum) for datum in data])

    def interdaily_stability_per_period(self, period="7D", freq="1h", verbose=False):
        r"""Interdaily stability per period

        The IS is calculated for each consecutive period found in the
        actigraphy recording.

        Parameters
        ----------
        period: str, optional
            Time period for the calculation of IS
            Default is '7D'.
        freq: str, optional
            Data resampling `frequency strings
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
        verbose: bool, optional
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.

        Returns
        -------
        isp: list of float

        Notes
        -----

        Periods are consecutive and all of the required duration. If the last
        consecutive period is shorter than required, the IS is not calculated
        for that period.
        """
        data = self.resample(data=self.activity, freq=freq)
        return interdaily_stability_per_period(
            data=data, period=period, verbose=verbose
        )

    def intradaily_variability(self, freq="1h"):
        r"""Intradaily variability

        The Intradaily Variability (IV) quantifies the variability of the
        activity recording. This variable thus measures the rest or activity
        fragmentation.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
            Default is '1h'.

        Returns
        -------
        iv: float

        Notes
        -----

        It is defined in ref [1]_:

        .. math::

            IV = \frac{c^{1h}}{d^{1h}}

        with:

        .. math::

            d^{1h} = \sum_{i}^{n}\frac{\left(x_{i}-\bar{x}\right)^{2}}{n}

        where :math:`x_{i}` is the number of active (counts higher than a
        predefined threshold) minutes during the :math:`i^{th}` period,
        :math:`\bar{x}` is the mean of all data and :math:`n` is the number of
        periods covered by the actigraphy data,

        and with:

        .. math::

            c^{1h} = \sum_{i}^{n-1} \frac{
                        \left( x_{i+1} - x_{i} \right)^{2}
                     }{n-1}

        References
        ----------

        .. [1] Witting W., Kwa I.H., Eikelenboom P., Mirmiran M., Swaab D.F.
               Alterations in the circadian rest–activity rhythm in aging and
               Alzheimer׳s disease. Biol Psychiatry. 1990;27:563–572.
        """
        if self.mask_inactivity:
            data = self.activity.where(self.mask > 0)
        else:
            data = self.activity

        return intradaily_variability(data)

    def average_intradaily_variability(self, freqs=None):
        r"""Average intradaily variability

        IVm [1]_ is the average of the IV values obtained with resampling
        periods divisors of 1440 between 1 and 60 min.

        Parameters
        ----------
        freq: str, optional
            Data resampling `frequency strings
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.

        Returns
        -------
        ivm: float

        Notes
        -----

        By default, the resampling periods are 1, 2, 3, 4, 5, 6, 8, 9, 10, 12,
        15, 16, 18, 20, 24, 30, 32, 36, 40, 45, 48 and 60 min.

        References
        ----------

        .. [1] Gonçalves, B. S., Cavalcanti, P. R., Tavares, G. R.,
               Campos, T. F., & Araujo, J. F. (2014). Nonparametric methods in
               actigraphy: An update. Sleep science (Sao Paulo, Brazil), 7(3),
               158-64.
        """
        if freqs is None:
            freqs = [
                "1min",
                "2min",
                "3Tmin",
                "4min",
                "5min",
                "6min",
                "8min",
                "9min",
                "10min",
                "12min",
                "15min",
                "16min",
                "18min",
                "20min",
                "24min",
                "30min",
                "32min",
                "36min",
                "40min",
                "45min",
                "48min",
                "60min",
            ]

        data = [self.resample(data=self.activity, freq=freq) for freq in freqs]

        return mean([intradaily_variability(datum) for datum in data])

    def intradaily_variability_per_period(self, period="7D", freq="1h", verbose=False):
        r"""Intradaily variability per period (method)

        The IV is calculated for each consecutive period found in the
        actigraphy recording.

        Parameters
        ----------
        period: str, optional
            Time period for the calculation of IS
            Default is '7D'.
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_.
            Default is '1h'.
        verbose: bool, optional
            If set to True, display the number of periods found in the activity
            recording, as well as the time not accounted for.
            Default is False.

        Returns
        -------
        ivp: list of float

        Notes
        -----

        Periods are consecutive and all of the required duration. If the last
        consecutive period is shorter than required, the IV is not calculated
        for that period.
        """
        data = self.resample(data=self.activity, freq=freq)
        return intradaily_variability(data=data, period=period, verbose=verbose)

    def pRA(self, threshold=0, start=None, period=None):
        r"""Rest->Activity transition probability distribution

        Conditional probability, pRA(t), that an individual would be
        resting at time (t+1) given that the individual had been continuously
        active for the preceding t epochs, defined in [1]_ as:

        .. math::
            pRA(t) = p(A|R_t) = \frac{N_t - N_{t+1}}{N_t}

        with :math:`N_t`, the total number of sequences of rest (i.e. activity
        below threshold) of duration :math:`t` or longer.

        Parameters
        ----------
        threshold: int
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
        start: str, optional
            If not None, the actigraphy recording is truncated to
            'start:start+period', each day. Start string format: 'HH:MM:SS'.
            Default is None
        period: str, optional
            Time period for the calculation of pRA.
            Default is None.

        Returns
        -------
        pra: pandas.core.series.Series
            Transition probabilities (pRA(t)), calculated for all t values.
        pra_weights: pandas.core.series.Series
            Weights are defined as the square root of the number of activity
            sequences contributing to each probability estimate.

        Notes
        -----

        pRA is corrected for discontinuities due to sparse data, as defined in
        [1]_.

        References
        ----------

        .. [1] Lim, A. S. P., Yu, L., Costa, M. D., Buchman, A. S.,
               Bennett, D. A., Leurgans, S. E., & Saper, C. B. (2011).
               Quantification of the Fragmentation of Rest-Activity Patterns in
               Elderly Individuals Using a State Transition Analysis. Sleep,
               34(11), 1569–1581. http://doi.org/10.5665/sleep.1400
        """

        # Restrict data range to period 'Start, Start+Period'
        if start is not None:
            end = _td_format(pd.Timedelta(start) + pd.Timedelta(period))

            data = self.binarize(data=self.activity, threshold=threshold).between_time(
                start, end
            )
        else:
            data = self.binarize(data=self.activity, threshold=threshold)

        # Rest->Activity transition probability:
        pRA, pRA_weights = _transition_prob(data, True)

        return pRA, pRA_weights

    def pAR(self, threshold, start=None, period=None):
        r"""Activity->Rest transition probability distribution

        Conditional probability, pAR(t), that an individual would be
        active at time (t+1) given that the individual had been continuously
        resting for the preceding t epochs, defined in [1]_ as:

        .. math::
            pAR(t) = p(R|A_t) = \frac{N_t - N_{t+1}}{N_t}

        with :math:`N_t`, the total number of sequences of activity (i.e.
        activity above threshold) of duration :math:`t` or longer.

        Parameters
        ----------
        threshold: int
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
        start: str, optional
            If not None, the actigraphy recording is truncated to
            'start:start+period', each day. Start string format: 'HH:MM:SS'.
            Default is None
        period: str, optional
            Time period for the calculation of pAR.
            Default is None.

        Returns
        -------
        par: pandas.core.series.Series
            Transition probabilities (pAR(t)), calculated for all t values.
        par_weights: pandas.core.series.Series
            Weights are defined as the square root of the number of activity
            sequences contributing to each probability estimate.

        Notes
        -----

        pAR is corrected for discontinuities due to sparse data, as defined in
        [1]_.

        References
        ----------

        .. [1] Lim, A. S. P., Yu, L., Costa, M. D., Buchman, A. S.,
               Bennett, D. A., Leurgans, S. E., & Saper, C. B. (2011).
               Quantification of the Fragmentation of Rest-Activity Patterns in
               Elderly Individuals Using a State Transition Analysis. Sleep,
               34(11), 1569–1581. http://doi.org/10.5665/sleep.1400
        """

        # Restrict data range to period 'Start, Start+Period'
        if start is not None:
            end = _td_format(pd.Timedelta(start) + pd.Timedelta(period))

            data = self.binarize(data=self.activity, threshold=threshold).between_time(
                start, end
            )
        else:
            data = self.binarize(data=self.activity, threshold=threshold)

        # Activity->Rest transition probability:
        pAR, pAR_weights = _transition_prob(data, False)

        return pAR, pAR_weights

    def kRA(
        self,
        threshold=0,
        start=None,
        period=None,
        frac=0.3,
        it=0,
        logit=False,
        freq=None,
        offset="15min",
    ):
        r"""Rest->Activity transition probability

        Weighted average value of pRA(t) within the constant regions, defined
        as the longest stretch within which the LOWESS curve varied by no more
        than 1 standard deviation of the pRA(t) curve [1]_.

        Parameters
        ----------
        threshold: int
            Above this threshold, data are classified as active (1) and as
            rest (0) otherwise.
        start: str, optional
            If not None, the actigraphy recording is truncated to
            'start:start+period', each day. Start string format: 'HH:MM:SS'.
            Special keywords ('AonT' or 'AoffT') are allowed. In this case, the
            start is set to the activity onset ('AonT') or offset ('AoffT')
            time derived from the daily profile. Cf sleep.AonT/AoffT functions
            for more informations.
            Default is None
        period: str, optional
            Time period for the calculation of pRA.
            Default is None.
        frac: float, optional
            Fraction of the data used when estimating each value.
            Default is 0.3.
        it: int, optional
            Number of residual-based reweightings to perform.
            Default is 0.
        logit: bool, optional
            If True, the kRA value is logit-transformed (ln(p/1-p)). Useful
            when kRA is used in a regression model.
            Default is False.
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_
            applied to the daily profile if start='AonT' or 'AoffT'.
            Default is None.
        offset: str, optional
            Time offset with respect to the activity onset and offset times
            used as start times.
            Default is '15min'.

        Returns
        -------
        kra: float

        References
        ----------

        .. [1] Lim, A. S. P., Yu, L., Costa, M. D., Buchman, A. S.,
               Bennett, D. A., Leurgans, S. E., & Saper, C. B. (2011).
               Quantification of the Fragmentation of Rest-Activity Patterns in
               Elderly Individuals Using a State Transition Analysis. Sleep,
               34(11), 1569–1581. http://doi.org/10.5665/sleep.1400
        """

        if start is not None and re.match(r"AonT|AoffT", start):
            aont = self.AonT(freq=freq, binarize=True, threshold=threshold)
            aofft = self.AoffT(freq=freq, binarize=True, threshold=threshold)
            offset = pd.Timedelta(offset)
            if start == "AonT":
                start_time = str(aont + offset).split(" ")[-1]
                period = str(
                    pd.Timedelta("24h") - ((aont + offset) - (aofft - offset))
                ).split(" ")[-1]
            elif start == "AoffT":
                start_time = str(aofft + offset).split(" ")[-1]
                period = str(
                    pd.Timedelta("24h") - ((aofft + offset) - (aont - offset))
                ).split(" ")[-1]
        else:
            start_time = start

        # Calculate the pRA probabilities and their weights.
        pRA, pRA_weights = self.pRA(threshold, start=start_time, period=period)

        # Fit the pRA distribution with a LOWESS and return mean value for
        # the constant region (i.e. the region where |pRA-lowess|<1SD)
        kRA = _transition_prob_sustain_region(pRA, pRA_weights, frac=frac, it=it)
        return np.log(kRA / (1 - kRA)) if logit else kRA

    def kAR(
        self,
        threshold,
        start=None,
        period=None,
        frac=0.3,
        it=0,
        logit=False,
        freq=None,
        offset="15min",
    ):
        r"""Rest->Activity transition probability

        Weighted average value of pAR(t) within the constant regions, defined
        as the longest stretch within which the LOWESS curve varied by no more
        than 1 standard deviation of the pAR(t) curve [1]_.

        Parameters
        ----------
        threshold: int
            Above this threshold, data are classified as active (1) and as
            rest (0) otherwise.
        start: str, optional
            If not None, the actigraphy recording is truncated to
            'start:start+period', each day. Start string format: 'HH:MM:SS'.
            Special keywords ('AonT' or 'AoffT') are allowed. In this case, the
            start is set to the activity onset ('AonT') or offset ('AoffT')
            time derived from the daily profile. Cf sleep.AonT/AoffT functions
            for more informations.
            Default is None
        period: str, optional
            Time period for the calculation of pRA.
            Default is None.
        frac: float
            Fraction of the data used when estimating each value.
            Default is 0.3.
        it: int
            Number of residual-based reweightings to perform.
            Default is 0.
        logit: bool, optional
            If True, the kRA value is logit-transformed (ln(p/1-p)). Useful
            when kRA is used in a regression model.
            Default is False.
        freq: str, optional
            Data resampling `frequency string
            <https://pandas.pydata.org/pandas-docs/stable/timeseries.html>`_
            applied to the daily profile if start='AonT' or 'AoffT'.
            Default is None.
        offset: str, optional
            Time offset with respect to the activity onset and offset times
            used as start times.
            Default is '15min'.

        Returns
        -------
        kar: float

        References
        ----------

        .. [1] Lim, A. S. P., Yu, L., Costa, M. D., Buchman, A. S.,
               Bennett, D. A., Leurgans, S. E., & Saper, C. B. (2011).
               Quantification of the Fragmentation of Rest-Activity Patterns in
               Elderly Individuals Using a State Transition Analysis. Sleep,
               34(11), 1569–1581. http://doi.org/10.5665/sleep.1400
        """

        if start is not None and re.match(r"AonT|AoffT", start):
            aont = self.AonT(freq=freq, binarize=True, threshold=threshold)
            aofft = self.AoffT(freq=freq, binarize=True, threshold=threshold)
            offset = pd.Timedelta(offset)
            if start == "AonT":
                start_time = str(aont + offset).split(" ")[-1]
                period = str(
                    pd.Timedelta("24h") - ((aont + offset) - (aofft - offset))
                ).split(" ")[-1]
            elif start == "AoffT":
                start_time = str(aofft + offset).split(" ")[-1]
                period = str(
                    pd.Timedelta("24h") - ((aofft + offset) - (aont - offset))
                ).split(" ")[-1]
        else:
            start_time = start

        # Calculate the pAR probabilities and their weights.
        pAR, pAR_weights = self.pAR(threshold, start=start_time, period=period)
        # Fit the pAR distribution with a LOWESS and return mean value for
        # the constant region (i.e. the region where |pAR-lowess|<1SD)
        kAR = _transition_prob_sustain_region(pAR, pAR_weights, frac=frac, it=it)
        return np.log(kAR / (1 - kAR)) if logit else kAR
