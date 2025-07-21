from circStudio.analysis.sleep import SleepDiary
from .mask import Mask
from ..analysis.tools import _data_processor
import numpy as np
import plotly.graph_objs as go


class Raw(Mask):
    """Base class for raw actigraphy data."""

    def __init__(
        self,
        df,
        period,
        frequency,
        activity,
        light,
        fpath=None,
        start_time=None,
        stop_time=None,
    ):
        self.df = df
        self.start_time = start_time
        self.stop_time = stop_time
        self.period = period
        self.frequency = frequency
        self.light = light
        self.activity = activity
        self.sleep_diary = None
        super().__init__(
            exclude_if_mask=True,
            mask_inactivity=False,
            inactivity_length=None,
            binarize=False,
            threshold=0,
            mask=None,
        )

    def plot(self, mode="activity", ts=None, log=False):
        """
        Plotting method for actigraphy data.

        Parameters
        ----------
        ts : str, optional
            In case the user wants to plot a time series (ts) other than light or activity,
            name of the vector to be used
        mode : str
            Either 'activity', 'light' or 'ts'. In case the user wants to plot a time series
            other than light or activity.
        log : bool
            Whether or not to log data (log_{10]+1).

        Returns
        -------
        go.Figure
            Plot of activity or light data.
        """
        if ts is not None:
            mode = None

        match mode:
            case "activity":
                # Define layout for activity plot
                layout = go.Layout(
                    title=f"Activity time series",
                    xaxis=dict(title="DateTime"),
                    yaxis=dict(title="Activity"),
                    showlegend=False,
                )
                if log:
                    # Draw lineplot corresponding to log activity data
                    return go.Figure(
                        data=go.Scatter(
                            x=self.activity.index.astype(str),
                            y=np.log10(self.activity + 1),
                        ),
                        layout=layout,
                    )
                else:
                    # Draw lineplot corresponding to the activity data
                    return go.Figure(
                        data=go.Scatter(
                            x=self.activity.index.astype(str), y=self.activity
                        ),
                        layout=layout,
                    )
            case "light":
                # Define layout for light plot
                layout = go.Layout(
                    title="Light time series",
                    xaxis=dict(title="Datetime"),
                    yaxis=dict(title="Light"),
                    showlegend=False,
                )
                if log:
                    # Draw interactive lineplot corresponding to the log light data
                    return go.Figure(
                        data=go.Scatter(
                            x=self.light.index.astype(str), y=np.log10(self.light + 1)
                        ),
                        layout=layout,
                    )
                else:
                    # Draw interactive lineplot corresponding to the light data
                    return go.Figure(
                        data=go.Scatter(
                            x=self.light.index.astype(str), y=self.light
                        ),
                        layout=layout,
                    )
            case _:
                if ts is not None and ts in self.df.columns:
                    layout = go.Layout(
                        title=f"{ts.lower().capitalize()} time series",
                        xaxis=dict(title="Datetime"),
                        yaxis=dict(title=f"{ts.lower().capitalize()}"),
                        showlegend=False,
                    )
                    if log:
                        return go.Figure(
                            data=go.Scatter(
                                x=self.df[ts].index.astype(str), y=np.log10(self.df[ts] + 1)
                            ),
                            layout=layout,
                        )
                    else:
                        return go.Figure(
                            data=go.Scatter(
                                x=self.df[ts].index.astype(str), y=self.df[ts]
                            ),
                            layout=layout,
                        )
                else:
                    print('No column was found with the specified name.')



    def length(self):
        r"""Number of activity data acquisition points"""
        return len(self.activity)

    def time_range(self):
        r"""Range (in days, hours, etc) of the activity data acquistion period"""
        return self.activity.index[-1] - self.activity.index[0]

    def duration(self):
        r"""Duration (in days, hours, etc) of the activity data acquistion period"""
        return self.frequency * self.length()

    def read_sleep_diary(
        self, input_fname, header_size=2, state_index=None, state_colour=None
    ):
        r"""Reader function for sleep diaries.

        Parameters
        ----------
        input_fname: str
            Path to the sleep diary file.
        header_size: int
            Header size (i.e. number of lines) of the sleep diary.
            Default is 2.
        state_index: dict
            Dictionnary of state's indices.
        state_color: dict
            Dictionnary of state's colours.
        """
        self.sleep_diary = SleepDiary(
            input_fname=input_fname,
            start_time=self.start_time,
            periods=self.length(),
            frequency=self.frequency,
            header_size=header_size,
            state_index=state_index,
            state_colour=state_colour,
        )
