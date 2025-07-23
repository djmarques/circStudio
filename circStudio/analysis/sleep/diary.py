import os
import pandas as pd
import pyexcel as pxl
import warnings
from circStudio.analysis.sleep import *


class SleepDiary:
    """Class for reading sleep diaries."""
    def __init__(self, input_fname, start_time, periods, frequency, header_size=2, state_index= None, state_colour= None):

        # Set the default state index and color if not specified by the user
        if state_index is None:
            state_index = {'ACTIVE': 2, 'NAP': 1, 'NIGHT': 0, 'NOWEAR': -1}

        if state_colour is None:
            state_colour = {'NAP': '#7bc043', 'NIGHT': '#d3d3d3', 'NOWEAR': '#ee4035'}

        # Get absolute file path
        input_fname = os.path.abspath(input_fname)

        sd_array = pxl.get_array(file_name=input_fname)

        self._name = sd_array[0][1]
        self._diary = pd.DataFrame(
            sd_array[header_size+1:],
            columns=sd_array[header_size]).astype({
                'TYPE': 'str',
                'START': 'datetime64[ns]',
                'END': 'datetime64[ns]'
            })

        # Inplace drop of useless columns
        self._diary.drop(
            columns=['DURATION (min)'],
            inplace=True,
            errors='ignore'
        )

        # Inplace drop of NA
        self._diary.dropna(inplace=True)

        self._state_index = state_index
        self._state_colour = state_colour

        # Create a time series with ACTIVE as default value.
        self._raw_data = pd.Series(
            data=self._state_index['ACTIVE'],
            index=pd.date_range(
                start_time,
                periods=periods,
                freq=frequency
            ),
            dtype=int
        )

        # Replace the default value with the ones found in the sleep diary.
        for index, row in self._diary.iterrows():
            self._raw_data[
                row['START']:row['END']
            ] = self._state_index[row['TYPE']]

        # Create a template shape to overlay over a plotly plot
        self._shaded_area = dict(
            type='rect',
            xref='x',
            yref='paper',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            fillcolor='',
            opacity=0.5,
            layer='below',
            line=dict(width=0)
        )

    @property
    def name(self):
        """The name of the subject."""
        return self._name

    @property
    def diary(self):
        """The dataframe containing the data found in the sleep diary."""
        return self._diary

    @property
    def state_index(self):
        """The indices assigned to the states found in the sleep diary."""
        return self._state_index

    @state_index.setter
    def state_index(self, value):
        self._state_index = value

    @property
    def state_colour(self):
        """The colours assigned to the states found in the sleep diary."""
        return self._state_colour

    @state_colour.setter
    def state_colour(self, value):
        self._state_colour = value

    @property
    def raw_data(self):
        """The time series related to the states found in the sleep diary."""
        return self._raw_data

    @property
    def shaded_area(self):
        """The template shape which can be overlaid over a plotly plot of the
        associated actimetry time series."""
        return self._shaded_area

    @shaded_area.setter
    def shaded_area(self, value):
        self._shaded_area = value

    def shapes(self):
        """ """
        shapes = []
        for index, row in self._diary.iterrows():
            shape = self._shaded_area.copy()
            shape['x0'] = row['START']
            shape['x1'] = row['END']
            shape['fillcolor'] = self._state_colour[row['TYPE']]
            shapes.append(shape)
        return shapes

    def summary(self):
        """ Returns a dataframe of summary statistics."""
        if 'DURATION' not in self._diary.columns:
            self._diary['DURATION'] = self._diary['END']\
                - self._diary['START']
        return self._diary.groupby(['TYPE'])['DURATION'].describe()

    def state_infos(self, state):
        """ Returns summary statistics for a given state

        Parameters
        ----------
        state: str
            State of interest
        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.
        """

        # Re-use the summary function
        summary = self.summary()

        # Verify that the state is present in the summary object
        if state not in summary.index:
            raise KeyError(
                "{} is not a valid state. Valid states are {}".format(
                    state, '" or "'.join(summary.index)
                )
            )

        # Access the summary object to get the mean
        mean = summary.loc[state, 'mean']
        # Access the summary object to get the std
        std = summary.loc[state, 'std']

        return mean, std

    def total_bed_time(self, state='NIGHT'):
        """ Returns the total in-bed time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NIGHT'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)

    def total_nap_time(self, state='NAP'):
        """ Returns the total nap time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NAP'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)

    def total_nowear_time(self, state='NOWEAR'):
        """ Returns the total 'no-wear' time

        Parameters
        ----------
        state : str, optional
            State of interest.
            Default is 'NOWEAR'.

        Returns
        -------
        mean: pd.Timedelta
            Mean duration of the required state.
        std: pd.Timedelta
            Standard deviation of the durations of the required state.

        """

        return self.state_infos(state)


    def sleep_efficiency(self, data):
        """
        Computes sleep efficiency as the average total sleep time, as classified by the Roenneberg algorithm,
        divided by the average total sleep time, as identified in the sleep diary.

        Parameters
        ----------
        data : pd.Series

        Returns
        -------
        float
            Sleep efficiency (decimal)

        """
        # Calculate average total sleep time (within the main sleep bout)
        avg_total_sleep_time = main_sleep_bouts(data=data)[1]

        # Calculate average total bedtime (from sleep diary)
        avg_total_bed_time = self.total_bed_time()[0]

        # If avg_total_bed_time is zero, do not return a result
        if avg_total_bed_time == 0:
            warnings.warn('Average total sleep time is 0.')
            return None

        # If avg_total_bed_time < avg_total_sleep_time
        if avg_total_sleep_time > avg_total_bed_time:
            warnings.warn('Average total sleep time is greater than average total sleep time.')
            return None

        return avg_total_sleep_time / avg_total_bed_time


    def sleep_onset_latency(self, data):
        """
        Computes sleep onset latency using the Roenneberg algorithm to predict sleep onset and
        the sleep diary to determine total bedtime.

        Parameters
        ----------
        data : pandas.Series
            Input data series with a DatetimeIndex, where the index specifies the time points and
            the values represent the input variable (e.g., activity, light). Time and value arrays
            are extracted from this series.

        Returns
        -------
        pd.Series
            Array containing sleep onset latency indexed by day of the recording.
        pd.Timedelta
            Mean sleep onset latency.

        """
        main_sleep_df = main_sleep_bouts(data=data)[0]
        diary_nights_df = self._diary[self._diary['TYPE'] == 'NIGHT']

        # Create an empty dictionary to store sleep_onset_latency (sol) values
        sol = {}

        # Iterate over the rows of the sleep diary corresponding to nighttime
        for _, row in diary_nights_df.iterrows():
            # Extract the date from the current row
            date = row['START'].date()

            # Identify matches between the sleep diary and detected periods of sleep
            matches = main_sleep_df[main_sleep_df['start_time'].dt.date == date]

            # If a match was found, then calculate the latency between bedtime and sleep onsets
            if not matches.empty:
                # Extract sleep onset
                sleep_onset = matches.iloc[0]['start_time']

                # Calculate the latency and store it in the sol dictionary
                latency = sleep_onset - row['START']
                sol[date] = latency
        # Typecast and return, sol to a pd.Series, along with the mean
        sol = pd.Series(sol)
        return pd.Series(sol), np.mean(sol)

