#############################################################################
# Copyright (c) 2022, Daylight Academy
# Author: Grégory Hammad
# Owner: Daylight Academy (https://daylight.academy)
# Maintainer: Grégory Hammad
# Email: gregory.hammad@uliege.be
# Status: development
#############################################################################
# The development of a module for analysing light exposure
# data was led and financially supported by members of the Daylight Academy
# Project “The role of daylight for humans” (led by Mirjam Münch, Manuel
# Spitschan). The module is part of the Human Light Exposure Database. For
# more information about the project, please see
# https://daylight.academy/projects/state-of-light-in-humans/.
#
# This module is also part of the circStudio software.
# circStudio is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# circStudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
############################################################################
import pandas as pd
import re
from scipy import signal
from ..activity.activity import _lmx
from ..activity.activity import _interdaily_stability
from ..activity.activity import _intradaily_variability
from ..tools.tools import _average_daily_activity
from ..tools.tools import _shift_time_axis

__all__ = ['Light', 'LightRecording']


class Light(object):
    """ Mixin Class """

    def _light_data(self, binarize=False, threshold=0, freq=None):
        # Binarize data using a given threshold
        if binarize:
            data = self.binarize(data=self.light, threshold=threshold)
        else:
            data = self.light

        print(data)

        # Resample and apply existing mask if enabled
        if freq is None:
            return self.resample(data=data)
        else:
            return self.resample(data=data, freq=freq)


    def average_daily_profile(self, freq='5min', cyclic=False, time_origin=None):
        r"""Average daily light profile

        Calculate the daily profile of light exposure. Data are averaged over
        all the days.

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
            Supported time string: 'HH:MM:SS'

        Returns
        -------
        raw : pandas.Series
            A Series containing the daily light profile with a 24h/48h index.
        """
        data = self._light_data(freq=freq)

        if time_origin is None:

            return _average_daily_activity(data, cyclic=cyclic)

        else:
            if cyclic is True:
                raise NotImplementedError(
                    'Setting a time origin while cyclic option is True is not '
                    'implemented yet.'
                )

            avgdaily = _average_daily_activity(data, cyclic=False)

            if isinstance(time_origin, str):
                # Regex pattern for HH:MM:SS time string
                pattern = re.compile(
                    r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$"
                )

                if pattern.match(time_origin):
                    time_origin = pd.Timedelta(time_origin)
                else:
                    raise ValueError(
                        'Time origin format ({}) not supported.\n'.format(
                            time_origin
                        ) + 'Supported format: HH:MM:SS.'
                    )

            elif not isinstance(time_origin, pd.Timedelta):
                raise ValueError(
                    'Time origin is neither a time string with a supported '
                    'format, nor a pd.Timedelta.'
                )

            # Round time origin to the required frequency
            time_origin = time_origin.round(data.index.freq)

            shift = int((pd.Timedelta('12h')-time_origin)/data.index.freq)

            return _shift_time_axis(avgdaily, shift)

    def average_daily_profile_auc(self, start_time=None, stop_time=None, time_origin=None, freq=None):
        r"""AUC of the average daily light profile

        Calculate the area under the curve of the daily profile of light
        exposure. Data are averaged over all the days.

        Parameters
        ----------
        start_time: str, optional
            If not set to None, compute AUC from start time.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, compute AUC until stop time.
            Supported time string: 'HH:MM:SS'
            Default is None.
        time_origin: str or pd.Timedelta, optional
            If not None, origin of the time axis for the daily profile.
            Original time bins are translated as time delta with respect to
            this new origin.
            Default is None
            Supported time string: 'HH:MM:SS'

        Returns
        -------
        auc : float
            Area under the curve.
        """
        data = self._light_data(freq=freq)

        # Compute average daily profile
        avgdaily = _average_daily_activity(data, cyclic=False)

        if time_origin is not None:

            if isinstance(time_origin, str):
                # Regex pattern for HH:MM:SS time string
                pattern = re.compile(
                    r"^([0-1]\d|2[0-3])(?::([0-5]\d))(?::([0-5]\d))$"
                )

                if pattern.match(time_origin):
                    time_origin = pd.Timedelta(time_origin)
                else:
                    raise ValueError(
                        'Time origin format ({}) not supported.\n'.format(
                            time_origin
                        ) + 'Supported format: HH:MM:SS.'
                    )

            elif not isinstance(time_origin, pd.Timedelta):
                raise ValueError(
                    'Time origin is neither a time string with a supported '
                    'format, nor a pd.Timedelta.'
                )

            # Round time origin to the required frequency
            time_origin = time_origin.round(data.index.freq)

            shift = int((pd.Timedelta('12h')-time_origin)/data.index.freq)

            avgdaily = _shift_time_axis(avgdaily, shift)

        # Restrict profile to start/stop times
        if start_time is not None:
            start_time = pd.Timedelta(start_time)
        if stop_time is not None:
            stop_time = pd.Timedelta(stop_time)

        # In order to avoid indexing with None, check for that too
        if start_time is not None or stop_time is not None:
            return avgdaily.loc[start_time:stop_time].sum()
        else:
            return avgdaily.sum()

    def _light_exposure(self, threshold=None, start_time=None, stop_time=None):
        r"""Light exposure

        Calculate the light exposure level and time

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.

        Returns
        -------
        masked_data : pandas.DataFrame
            A DataFrame where the original data are set to Nan if below
            threshold and/or outside time window.
        """
        if threshold is not None:
            data_mask = self.light.mask(self.light < threshold)
        else:
            data_mask = self.light

        if start_time is stop_time is None:
            return data_mask
        elif (start_time is None) or (stop_time is None):
            raise ValueError('Both start and stop times have to be specified, if any.')
        else:
            return data_mask.between_time(start_time=start_time, end_time=stop_time, include_end=False)

    def light_exposure_level(self, threshold=None, start_time=None, stop_time=None, agg='mean'):
        r"""Light exposure level

        Calculate the aggregated (mean, median, etc) light exposure level
        per epoch.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        agg: str, optional
            Aggregating function used to summarize exposure levels.
            Available functions: 'mean', 'median', 'std', etc.
            Default is 'mean'.

        Returns
        -------
        levels : pd.Series
            A pandas Series with aggreagted light exposure levels per channel
        """
        light_exposure = self._light_exposure(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time
        )

        levels = getattr(light_exposure, agg)

        return levels()

    def summary_statistics_per_time_bin(
        self,
        bins='24h',
        agg_func=['mean', 'median', 'sum', 'std', 'min', 'max']
    ):
        r"""Summary statistics.

        Calculate summary statistics (ex: mean, median, etc) according to a
        user-defined (regular or arbitrary) binning.

        Parameters
        ----------
        bins: str or list of tuples, optional
            If set to a string, bins is used to define a regular binning where
            every bin is of length "bins". Ex: "2h".
            Otherwise, the list of 2-tuples is used to define an arbitrary
            binning. Ex: \[('2000-01-01 00:00:00','2000-01-01 11:59:00')\].
            Default is '24h'.
        agg_func: list, optional
            List of aggregation functions to be used on every bin.
            Default is \['mean', 'median', 'sum', 'std', 'min', 'max'\].

        Returns
        -------
        ss : pd.DataFrame
            A pandas DataFrame with summary statistics per channel.
        """
        if isinstance(bins, str):
            summary_stats = self.light.resample(bins).agg(agg_func)
        elif isinstance(bins, list):
            df_col = []
            for idx, (start, end) in enumerate(bins):
                df_bins = self.light.loc[start:end, :].apply(
                    agg_func
                ).pivot_table(columns=agg_func)
                channels = {}
                for ch in df_bins.index:
                    channels[ch] = df_bins.loc[df_bins.index == ch]
                    channels[ch] = channels[ch].rename(
                        index={ch: idx},
                        inplace=False
                    )
                    channels[ch] = channels[ch].loc[:, agg_func]
                df_col.append(
                    pd.concat(
                        channels,
                        axis=1
                    )
                )
            summary_stats = pd.concat(df_col)

        return summary_stats

    def time_above_threshold(
        self, threshold=None, start_time=None, stop_time=None, oformat=None
    ):
        r"""Time above light threshold.

        Calculate the total light exposure time above the threshold.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        oformat: str, optional
            Output format. Available formats: 'minute' or 'timedelta'.
            If set to 'minute', the result is in number of minutes.
            If set to 'timedelta', the result is a pd.Timedelta.
            If set to None, the result is in number of epochs.
            Default is None.

        Returns
        -------
        tat : pd.Series
            A pandas Series with aggreagted light exposure levels per channel
        """
        available_formats = [None, 'minute', 'timedelta']
        if oformat not in available_formats:
            raise ValueError(
                'Specified output format ({}) not supported. '.format(oformat)
                + 'Available formats are: {}'.format(str(available_formats))
            )

        light_exposure_counts = self._light_exposure(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time
        ).count()

        if oformat == 'minute':
            tat = light_exposure_counts * \
                pd.Timedelta(self.light.index.freq)/pd.Timedelta('1min')
        elif oformat == 'timedelta':
            tat = light_exposure_counts * pd.Timedelta(self.light.index.freq)
        else:
            tat = light_exposure_counts

        return tat

    def time_above_threshold_by_period(
        self, threshold=None, start_time=None, stop_time=None, oformat=None
    ):
        r"""Time above light threshold (per day).

        Calculate the total light exposure time above the threshold,
        per calendar day.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.
        start_time: str, optional
            If not set to None, discard data before start time,
            on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        stop_time: str, optional
            If not set to None, discard data after stop time, on a daily basis.
            Supported time string: 'HH:MM:SS'
            Default is None.
        oformat: str, optional
            Output format. Available formats: 'minute' or 'timedelta'.
            If set to 'minute', the result is in number of minutes.
            If set to 'timedelta', the result is a pd.Timedelta.
            If set to None, the result is in number of epochs.
            Default is None.

        Returns
        -------
        tatp : pd.DataFrame
            A pandas DataFrame with aggreagted light exposure levels
            per channel and per day.
        """
        available_formats = [None, 'minute', 'timedelta']
        if oformat not in available_formats:
            raise ValueError(
                'Specified output format ({}) not supported. '.format(oformat)
                + 'Available formats are: {}'.format(str(available_formats))
            )

        light_exposure_counts_per_day = self._light_exposure(
            threshold=threshold,
            start_time=start_time,
            stop_time=stop_time
        ).groupby(self.light.index.date).count()

        if oformat == 'minute':
            tatp = light_exposure_counts_per_day * \
                pd.Timedelta(self.light.index.freq)/pd.Timedelta('1min')
        elif oformat == 'timedelta':
            tatp = light_exposure_counts_per_day * pd.Timedelta(self.light.index.freq)
        else:
            tatp = light_exposure_counts_per_day

        return tatp

    def values_above_threshold(self, threshold=None):
        r"""Values above light threshold.

        Returns the light exposure values above the threshold.

        Parameters
        ----------
        threshold: float, optional
            If not set to None, discard data below threshold before computing
            exposure levels.
            Default is None.

        Returns
        -------
        vat : pd.Series
            A pandas Series with light exposure levels per channel
        """

        return self._light_exposure(
            threshold=threshold,
            start_time=None,
            stop_time=None
        )

    @classmethod
    def get_time_barycentre(cls, data):
        # Normalize each epoch to midnight.
        Y_j = data.index-data.index.normalize()
        # Convert to indices.
        Y_j /= pd.Timedelta(data.index.freq)
        # Compute barycentre
        bc = data.multiply(Y_j, axis=0).sum() / data.sum()

        return bc

    def mean_light_timing(self, threshold, freq=None):
        r"""Mean light timing.

        Mean light timing above threshold, MLiT^C.


        Parameters
        ----------
        threshold: float
            Threshold value.

        Returns
        -------
        MLiT : pd.DataFrame
            A pandas DataFrame with MLiT^C per channel.

        Notes
        -----

        The MLiT variable is defined in ref [1]_:

        .. math::

            MLiT^C = \frac{\sum_{j}^{m}\sum_{k}^{n} j\times I^{C}_{jk}}{
            \sum_{j}^{m}\sum_{k}^{n} I^{C}_{jk}}

        where :math:`I^{C}_{jk}` is equal to 1 if the light level is higher
        than the threshold C, m is the total number of epochs per day and n is
        the number of days covered by the data.

        References
        ----------

        .. [1] Reid K.J., Santostasi G., Baron K.G., Wilson J., Kang J.,
               Zee P.C., Timing and Intensity of Light Correlate with Body
               Weight in Adults. PLoS ONE 9(4): e92251.
               https://doi.org/10.1371/journal.pone.0092251

        """
        data = self._light_data(freq=freq)

        # Binarized data and convert to float in order to handle 'DivideByZero'
        I_jk = self.binarize(data=data, threshold=threshold).astype('float64')

        MLiT = self.get_time_barycentre(I_jk)

        # Scaling factor: MLiT is now expressed in minutes since midnight.
        MLiT /= (pd.Timedelta('1min')/I_jk.index.freq)

        return MLiT

    def mean_light_timing_by_period(self, threshold, freq=None):
        r"""Mean light timing per day.

        Mean light timing above threshold, MLiT^C, per calendar day.


        Parameters
        ----------
        threshold: float
            Threshold value.

        Returns
        -------
        MLiTp : pd.DataFrame
            A pandas DataFrame with MLiT^C per channel and per day.

        Notes
        -----

        The MLiT variable is defined in ref [1]_:

        .. math::

            MLiT^C = \frac{\sum_{j}^{m}\sum_{k}^{n} j\times I^{C}_{jk}}{
            \sum_{j}^{m}\sum_{k}^{n} I^{C}_{jk}}

        where :math:`I^{C}_{jk}` is equal to 1 if the light level is higher
        than the threshold C, m is the total number of epochs per day and n is
        the number of days covered by the data.

        References
        ----------

        .. [1] Reid K.J., Santostasi G., Baron K.G., Wilson J., Kang J.,
               Zee P.C., Timing and Intensity of Light Correlate with Body
               Weight in Adults. PLoS ONE 9(4): e92251.
               https://doi.org/10.1371/journal.pone.0092251

        """
        data = self._light_data(freq=freq)

        # Binarized data and convert to float in order to handle 'DivideByZero'
        I_jk = self.binarize(data=data, threshold=threshold).astype('float64')

        # Group data per day:
        MLiTp = I_jk.groupby(I_jk.index.date).apply(self.get_time_barycentre)

        # Scaling factor: MLiT is now expressed in minutes since midnight.
        MLiTp /= (pd.Timedelta('1min')/I_jk.index.freq)

        return MLiTp

    def get_light_extremum(self, extremum, freq=None):
        r"""Light extremum.

        Return the index and the value of the requested extremum (min or max).

        Parameters
        ----------
        extremum: str
            Name of the extremum.
            Available: 'min' or 'max'.

        Returns
        -------
        ext : pd.DataFrame
            A pandas DataFrame with extremum info per channel.
        """
        data = self._light_data(freq=freq)

        # Return either the maximum or minimum, as well as the respective timestamp
        if extremum == 'max':
            return data.idxmax(), data.max()
        elif extremum == 'min':
            return data.idxmin(), data.min()
        else:
            raise ValueError('Extremum must be "min" or "max"')


    def LMX(self, length='5h', lowest=True, freq=None):
        r"""Least or Most light period of length X

        Onset and mean hourly light exposure levels during the X least or most
        bright hours of the day.

        Parameters
        ----------
        length: str, optional
            Period length.
            Default is '5h'.
        lowest: bool, optional
            If lowest is set to True, the period of least light exposure is
            considered. Otherwise, consider the period of most light exposure.
            Default is True.

        Returns
        -------
        lmx_t, lmx: (pd.Timedelta, float)
            Onset and mean hourly light exposure level.

        Notes
        -----

        The LMX variable is derived from the L5 and M10 defined in [1]_ as the
        mean hourly activity levels during the 5/10 least/most active hours.

        References
        ----------

        .. [1] Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F.
               (1997). Long-Term Fitness Training Improves the Circadian
               Rest-Activity Rhythm in Healthy Elderly Males.
               Journal of Biological Rhythms, 12(2), 146–156.
               http://doi.org/10.1177/074873049701200206

        """
        data = self._light_data(freq=freq)

        # Calculate time of LMX and the value of LMX
        lmx_ts, lmx = _lmx(data, length, lowest=lowest)

        # Return these values back to the user
        return lmx_ts, lmx


    def _RAR(self, rar_func, rar_name, binarize=False, threshold=0, freq=None):
        r""" Generic rest-activity rhythm (RAR) function

        Apply a generic RAR function to the light data.
        """
        data = self._light_data(binarize=binarize, threshold=threshold, freq=freq)

        # Apply the RAR function
        return rar_func(data)


    def light_interdaily_stability(self, binarize=False, threshold=0):
        r"""Interdaily stability

        The Interdaily stability (IS) quantifies the repeatibilty of the
        daily light exposure pattern over each day contained in the activity
        recording.

        Parameters
        ----------
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is False.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 0.

        Returns
        -------
        is : pd.DataFrame
            A pandas DataFrame with IS per channel.


        Notes
        -----

        This variable is derived from the original IS variable defined in
        ref [1]_ as:

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

        For the record, this is the 24h value from the chi-square periodogram
        (Sokolove and Bushel, 1978).

        References
        ----------

        .. [1] Witting W., Kwa I.H., Eikelenboom P., Mirmiran M., Swaab D.F.
               Alterations in the circadian rest–activity rhythm in aging and
               Alzheimer׳s disease. Biol Psychiatry. 1990;27:563–572.
        """

        return self._RAR(
            _interdaily_stability,
            'interdaily_stability',
            binarize=binarize,
            threshold=threshold
        )

    def light_intradaily_variability(self, binarize=False, threshold=0):
        r"""Intradaily variability

        The Intradaily Variability (IV) quantifies the variability of the
        light exposure pattern.

        Parameters
        ----------
        binarize: bool, optional
            If set to True, the data are binarized.
            Default is False.
        threshold: int, optional
            If binarize is set to True, data above this threshold are set to 1
            and to 0 otherwise.
            Default is 4.

        Returns
        -------
        iv: pd.DataFrame
            A pandas DataFrame with IV per channel.

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
        return self._RAR(
            _intradaily_variability,
            'intradaily_variability',
            binarize=binarize,
            threshold=threshold
        )
