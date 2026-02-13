import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


class Light:
    """
    Create a synthetic light schedule.

    Internal variables:
    -------------------
    light_vector : np.ndarray
        Light intensity samples (lux).
    time_vector : np.ndarray
        Time in hours since start (float).

    Time axis
    ---------
    index : pd.DatetimeIndex | None
        If provided, this is the time axis that will be used.
        If not provided, you can build it from start + sampling interval
    """

    def __init__(self, time_vector=None, light_vector=None, index=None):
        self.light_vector = light_vector
        self.time_vector = time_vector
        self.index = index
        self.light_series = None
        self.gradient = None
        self.intercept = None

    @property
    def synthetic_light(self) -> pd.Series:
        """
        Return the synthetic light schedule as a pandas Series.

        The returned Series is indexed by the object's DatetimeIndex and
        contains the light intensity values (in lux).

        Returns
        -------
        pd.Series
            Light intensity time series with a DatetimeIndex.

        Raises
        ------
        ValueError
            If no DatetimeIndex has been set. Call `create(..., start=...)`
            or `with_datetime_index(...)` first.
        """
        if self.index is None:
            raise ValueError(
                "No DatetimeIndex set. Generate one with create()",
                "or with_datetime_index().",
            )
        return pd.Series(self.light_vector, index=self.index, name="synthetic_light")

    def with_datetime_index(
        self,
        start: str | pd.Timestamp,
        timezone: str | None = None,
        name: str = "light",
    ) -> pd.Series:
        """
        Create a DatetimeIndex using current sampling interval and then return a series.

        Parameters
        ----------
        start : str | pd.Timestamp
            Start timestamp for the light schedule.

        timezone : str | None
            Timezone of the light schedule.

        name : str
            Name of the light schedule. Default is "Light".

        Returns
        -------
        pd.Series
            Light schedule as a pandas Series with a Datetime index.
        """
        start = pd.Timestamp(start)

        # Check if a timezone (tz) has been provided by the user
        if timezone is not None:
            # Case in which there is no tz information
            if start.tzinfo is None:
                # Use the user-specified tz to attach a timezone
                start = start.tz_localize(timezone)
            # Case in which there is some tz information
            else:
                # Convert to newly specified timezone
                start = start.tz_convert(timezone)

        # Calculate time step (assuming regularly spaced data)
        dt_hours = float(self.time_vector[1] - self.time_vector[0])

        # Convert time step to timedelta object, measured in hours
        dt = pd.to_timedelta(dt_hours, unit="h")

        self.index = pd.date_range(start=start, periods=len(self.light_vector), freq=dt)

        light_series = pd.Series(self.light_vector, index=self.index, name=name)

        # Set attribute light_series and return it back to the user
        self.light_series = light_series
        return light_series

    def _dt_minutes(self) -> float:
        """Sampling interval in minutes (robust for fractional-hour time vectors)."""
        dt_hours = float(self.time_vector[1] - self.time_vector[0])
        return dt_hours * 60.0

    def create_mask(self, min_length_minutes, max_length_minutes):
        # Sampling interval calculation
        sampling_interval = self._dt_minutes()
        if sampling_interval <= 0:
            raise ValueError("Non-positive sampling interval detected.")

        # Convert minutes to number of indices based on sampling interval
        min_length_indices = int(round(min_length_minutes / sampling_interval))
        max_length_indices = int(round(max_length_minutes / sampling_interval))

        # Identify positions where the light vector is zero
        is_zero = self.light_vector == 0

        # Detect changes from zero to non-zero and vice versa
        changes = np.diff(is_zero.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        # Handle edge cases where the light vector starts or ends with zeros
        if is_zero[0]:
            # Case in which light vector starts with zero
            starts = np.insert(starts, 0, 0)
        if is_zero[-1]:
            # Case in which light vector ends with zero
            ends = np.append(ends, len(self.light_vector))

        # Store the durations for zero intervals on the light vector
        durations = ends - starts

        # Initialize a boolean mask of the same size as the light vector
        mask = np.zeros_like(self.light_vector, dtype=bool)

        # Filter which zero intervals are relevant based on specified min and max length
        for start, end, duration in zip(starts, ends, durations):
            if min_length_indices <= duration <= max_length_indices:
                # Set the mask to true for the values to be masked
                mask[start:end] = True

        return mask

    def interpolate_mask(self, mask):
        valid_idxs = np.where(mask == False)[0]
        spurious_idxs = np.where(mask == True)[0]

        # Convert mask to integers and get differences
        mask_diff = np.diff(mask.astype(int))

        # Start indices (mask goes from 0 to 1)
        start_idxs = np.where(mask_diff == 1)[0]

        # End indices (mask goes from 1 to 0)
        end_idxs = np.where(mask_diff == -1)[0]

        # Loop through each missing segment
        for start, end in zip(start_idxs, end_idxs):
            # Define the region to fill
            fill_range = np.arange(start, end + 1)

            # Get the points just before and after the missing region
            left_bound = start + 1
            right_bound = end + 2

            # Now interpolate between these boundary points
            xp = [left_bound, right_bound]
            fp = [self.light_vector[left_bound], self.light_vector[right_bound]]

            # Interpolate across the fill_range
            interpolated_values = np.interp(fill_range, xp, fp)

            # Replace the original data in the missing region
            self.light_vector[start + 1 : end + 2] = interpolated_values

    def downsample(self, factor):
        """
        Adjust the frequency at which light intensity values are sampled. The original
        data is sampled at a specific interval (e.g., 1 minute, 1 hour). When you provide
        a downsampling interval, this method increases the time between each sampled point
        by a factor of the given interval.

        Parameters
        ----------
        factor : float
            Downsampling factor in hours.
        """
        # Calculate the downsampling factor based on the provided downsampling factor and time step
        downsampling_factor = factor // (self.time_vector[1] - self.time_vector[0])

        # Calculate remainder to check if light vector length is divisible by downsampling factor
        remainder = len(self.light_vector) % downsampling_factor

        # Truncate the excess data to fit evenly, if not divisible
        if remainder != 0:
            self.time_vector = self.time_vector[:-remainder]
            self.light_vector = self.light_vector[:-remainder]

        # Repopulate the time and light vectors with the downsampled data
        self.time_vector = self.time_vector[::downsampling_factor]
        self.light_vector = self.light_vector.reshape(-1, downsampling_factor).mean(
            axis=1
        )

    @classmethod
    def create(
        cls,
        total_days=50,
        light_on_hours=16,
        bins_per_hour=6,
        schedule_starts_at=0,
        low=0,
        high=1000,
        start: str | pd.Timestamp | None = None,
        timezone: str | None = None,
    ):
        """
        Create a synthetic light schedule.

        Parameters
        ----------
        total_days : int
            Total number of days to include in the synthetic light schedule.
        light_on_hours : int
            How many hours per day with light on.
        bins_per_hour : int
            Epoch resolution (e.g., 10 means 6-min bins).
        schedule_starts_at : int
            Hour-of-day when the light-on block starts.
        low, high : float
            Lux bounds for the light-on block (uniform random if low!=high).
        start : str | pd.Timestamp | None
            If provided, attach a DatetimeIndex beginning at this timestamp.
        timezone : str | None
            Optional timezone (e.g., "Europe/Lisbon").

        Returns
        -------
        Light
            New synthetic light instance.
        """
        # Check if the number of bins for the on/off period is valid
        if not (0 <= light_on_hours <= 24):
            raise ValueError(
                "light_on_hours and light_off_hours must be between 0 and 24."
            )

        # Calculate the number of bins for light-on and light-off periods
        light_on_bins = int(light_on_hours * bins_per_hour)
        light_off_bins = int((24 - light_on_hours) * bins_per_hour)

        if low == high:
            light_on_variation = np.full(shape=(1, light_on_bins), fill_value=low)[0]
        else:
            # Instantiate a new generator
            rng = np.random.default_rng()

            # Generate random light intensity values for the light-on period
            light_on_variation = rng.uniform(low=low, high=high, size=light_on_bins)

        # Create a daily schedule with light-on and light-off periods
        daily_schedule = np.concatenate([light_on_variation, np.zeros(light_off_bins)])

        # Determine the start position for the light-on period
        start_bin = int(round(schedule_starts_at * bins_per_hour))
        start_bin %= 24 * bins_per_hour

        # Shift the schedule to the specified start time
        shifted_schedule = np.roll(daily_schedule, start_bin)

        # Repeat the daily schedule for the total number of days
        light_vector = np.tile(shifted_schedule, total_days)

        # Calculate the time step and generate the time vector
        dt = 1 / bins_per_hour
        time_vector = np.arange(len(light_vector)) * dt

        obj = cls(time_vector=time_vector, light_vector=light_vector, index=None)
        if start is not None:
            obj.with_datetime_index(start, timezone if timezone else None)

        return obj

    def __add__(self, other, shift=0):
        """
        Add two light schedules, optionally with a time shift.

        Parameters
        ----------
        other : Light
            Light schedule to add.
        shift : int, optional
            Time shift for the second schedule being added.

        Returns
        -------
        Light
            A new synthetic light instance representing the combined schedule
        """
        # Calculate bins per hour for the current schedule
        bins_per_hour = int(1 / (self.time_vector[1] - self.time_vector[0]))

        # Convert the time shift from hours to bins
        shift *= bins_per_hour
        shift = int(round(shift))

        # Merge the time vectors of both schedules (set union)
        common_time_vector = np.union1d(self.time_vector, other.time_vector)

        # Interpolate both light vectors to the common time vector
        f_self = interp1d(
            self.time_vector,
            self.light_vector,
            kind="previous",
            bounds_error=False,
            fill_value=0,
        )

        f_other = interp1d(
            other.time_vector,
            other.light_vector,
            kind="previous",
            bounds_error=False,
            fill_value=0,
        )

        # Add the light vectors with the specified time shift
        combined_light_vector = f_self(common_time_vector) + np.roll(
            f_other(common_time_vector), shift
        )

        # Return a new Light instance with the combined schedule
        return Light(common_time_vector, combined_light_vector)

    def __sub__(self, other, shift=0):
        """
        Subtract a light schedule, optionally with a time shift

        Parameters
        ----------
        other : Light
            Light schedule to add.
        shift : int, optional
            Time shift for the second schedule being added.

        Returns
        -------
        Light
            A new synthetic light instance representing the combined schedule
        """
        # Calculate bins per hour for the current schedule
        bins_per_hour = int(1 / (self.time_vector[1] - self.time_vector[0]))

        # Convert the time shift from hours to bins
        shift *= bins_per_hour
        shift = int(round(shift))

        # Merge the time vectors of both schedules
        common_time_vector = np.union1d(self.time_vector, other.time_vector)

        # Interpolate both light vectors to the common time vector
        f_self = interp1d(
            self.time_vector,
            self.light_vector,
            kind="previous",
            bounds_error=False,
            fill_value=0,
        )

        f_other = interp1d(
            other.time_vector,
            other.light_vector,
            kind="previous",
            bounds_error=False,
            fill_value=0,
        )

        # Subtract the light vectors and ensure non-negative values
        combined_light_vector = f_self(common_time_vector) - np.roll(
            f_other(common_time_vector), shift
        )
        combined_light_vector = np.clip(combined_light_vector, a_min=0, a_max=None)

        # Return a new Light instance with the resulting schedule
        return Light(common_time_vector, combined_light_vector)

    def __mul__(self, scalar):
        """
        Multiply light schedule by a scalar factor.

        Parameters
        ----------
        scalar : float
            Number that will be multiplied by the light intensities.

        Returns
        -------
        Light
            A new synthetic light instance representing the scaled schedule
        """
        return Light(self.time_vector, self.light_vector * scalar)

    def __truediv__(self, scalar):
        """
        Divide the light schedule by a scalar factor

        Parameters
        ----------
        scalar : float
            Number by which the light intensities will be divided.

        Returns
        -------
        Light
            A new synthetic light instance representing the divided schedule
        """
        return Light(self.time_vector, self.light_vector / scalar)

    def __str__(self):
        """
        Obtain a string containing the light intensity for each point in time
        """
        # Unicode subscript characters for time digits
        subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

        # Convert the index for each time point to subscript characters
        def _to_subscript(index):
            return "".join(subscript_digits[int(digit)] for digit in str(index))

        # Return a string containing the light intensity for each point in time
        paired_vectors = {
            time: light for time, light in zip(self.time_vector, self.light_vector)
        }
        return "\n".join(
            [
                f"t{_to_subscript(i)} = {time:.2f}, light = {light:.2f}"
                for i, (time, light) in enumerate(paired_vectors.items())
            ]
        )

    def train_activity_to_lux_model(
            self,
            activity_data: pd.Series=None,
            light_data: pd.Series=None,
            mode: str = 'linear',
            q: float = 0.95,
            ref_lux: float  | None = None,
    ) -> tuple[float, float]:
        """
        Train a mapping from activity counts to light intensity (lux).

        Modes
        -----
        linear:
            Fits y = gradient*x + intercept using non-linear least squares (curve_fit).

        quantile:
            Calibrates a linear mapping using quantiles. By default, maps the activity
            q-quantile to the light q-quantile (distribution-matching). If `ref_lux` is
            provided, maps the activity q-quantile to `ref_lux` instead.

        Parameters
        ----------
        activity_data : pandas.Series
            Time-indexed activity signal.
        light_data : pandas.Series
            Time-indexed light intensity signal in lux.
        q : float, optional
            Quantile in (0,1) used for quantile calibration. Default is 0.95.
            Only relevant if `mode` is 'quantile'.
        ref_lux : float or None, optional
            Reference light intensity in lux. If provided and mode is 'quantile',
            it maps qx(q) -> ref_lux. If None, it maps Qx(q) -> Qy(q).
        mode : str, optional
            Mode of linear regression model. Default is 'linear'.

        Returns
        -------
        tuple[float, float]
            (gradient, intercept) parameters of the fitted linear model.

        Notes
        -----
        - The model assumes a linear relationship between activity and light.
        - Both series should be aligned in time (same sampling resolution).
        - Missing values are automatically removed prior to fitting.
        - The returned parameters can later be used to estimate light
        from new activity data.
        """
        if activity_data is None or light_data is None:
            raise ValueError("activity_data and light_data must be provided.")

        # Concatenate series to create a combined dataframe with missing values removed
        df = pd.concat(
            [activity_data.rename('activity'), light_data.rename('light')],
            axis=1
        ).dropna()

        # Raise error in case of complete misalignment between activity and light series
        if df.empty:
            raise ValueError("No overlap between activity data and light data")

        # Convert to numpy arrays
        x = df['activity'].to_numpy(dtype=float)
        y = df['light'].to_numpy(dtype=float)

        gradient, intercept = None, None
        mode = mode.lower()
        match mode:
            case "linear":
                # Define linear function for the mapping
                def _light(activity, delta, c):
                    return delta * activity + c

                # Extract optimal gradient and intercept
                (gradient, intercept), _ = curve_fit(_light, x, y, bounds=([0, 0], [np.inf, np.inf]))

            case "quantile":
                if not (0 < q < 1):
                    raise ValueError("q must be between 0 and 1 (exclusive).")

                # Activity quantile (qx)
                qx = float(np.nanquantile(x,q))

                # Check if the obtained value for qx is finite and positive
                if not np.isfinite(qx) or qx <=0:
                    raise ValueError(f"Activity quantile Qx({q}) is not > 0; cannot calibrate.")

                # Light quantile (qy, our target)
                if ref_lux is not None:
                    # Set it to ref_lux if specified by the user
                    qy = float(ref_lux)
                else:
                    # Calculate the quantile if no ref_lux is provided
                    qy = np.nanquantile(y, q)

                # Check if the obtained value for qy is finite and positive
                if not np.isfinite(qy) or qy < 0:
                    raise ValueError("Light quantile is invalid.")

                # Calculate the gradient based on quantile ratio
                gradient = qy/qx

                # Set intercept to zero
                intercept = 0

            case _:
                raise ValueError("Unknown mode. Choose 'linear' or 'quantile'")

        # Set the gradient and intercept attributes
        self.gradient = float(gradient)
        self.intercept = float(intercept)

        # Return the output
        return float(gradient), float(intercept)

    def convert_activity_to_lux(self, activity):
        # Check if a gradient or intercept were previously calculated
        if self.gradient is None or self.intercept is None:
            raise ValueError("Parameters were not provided. Please run train_activity_to_lux_model.")

        # Check if activity series was provided
        if activity is None:
            raise ValueError("Activity trace was not provided.")

        # Compute linear model and return it
        self.light_series = self.gradient * activity + self.intercept
        return self.light_series


    def actogram(
        self,
        size=(12, 6),
        title="Actogram",
        legend_loc="upper left",
        legend_fontsize=10,
        activity_color="#404040",
        inactivity_color="#DCDCDC",
        midline_color="red",
        show = False,
    ):
        """
        Double-plotted actogram of the synthetic light schedule.

        The light intensity values are reshaped into daily segments
        and displayed as a heatmap, where darker colors indicate higher
        light intensity (activity period) and lighter colors indicate
        zero or low light intensity.

        Parameters
        ----------
        size : tuple of float, optional
            Figure size in inches (width, height). Defaults to (12, 6).

        title : str, optional
            Title of the plot. Defaults is "Actogram".

        legend_loc : str, optional
            Location of the legend in matplotlib format. Defaults to "upper left".

        legend_fontsize : int, optional
            Font size of the legend. Defaults to 10.

        activity_color : str, optional
            Color used to represent periods with light exposure
            (non-zero light intensity). Default is "#404040".

        inactivity_color : str, optional
            Color used to represent periods without light exposure
            (zero light intensity). Default is "#DCDCDC".

        midline_color : str, optional
            Color of the vertical dashed line separating the two 24-hour
            segments in the double-plotted actogram. Default is "red".

        show : bool, optional
            Whether to show the plot. Defaults is False.

        Notes
        -----
        - This method assumes regularly sampled time data.
        - The sampling interval is inferred from `self.time_vector`.
        - The total duration must correspond to an integer number of days.
        - The function displays the figure using `plt

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.pyplot.Axes
            The double-plotted actogram plot.

        """
        # Determine the number of bins per day
        bins_per_day = int(24 / np.diff(self.time_vector)[0])

        # Determine the total number of days
        total_time = self.time_vector[-1] - self.time_vector[0]
        num_days = int(np.ceil(total_time / 24))

        # Reshape the light vector for the actogram
        light_matrix = self.light_vector.reshape(num_days, bins_per_day)

        # Create double-plotted data
        double_plot_matrix = np.tile(light_matrix, 2)

        # Create figure to store the heatmap
        fig = plt.figure(figsize=size)

        # Get current axes, the current axes object active in this figure
        ax = plt.gca()

        # Create a custom color map
        custom_cmap = LinearSegmentedColormap.from_list(
            "pastel", [inactivity_color, activity_color]
        )

        # Create the heatmap
        sns.heatmap(
            double_plot_matrix,
            cmap=custom_cmap,
            cbar=False,
            yticklabels=np.arange(1, light_matrix.shape[0] + 1),
            ax=ax
        )

        # Add horizontal lines to separate rows (days)
        for i in range(0, double_plot_matrix.shape[0] + 1):
            plt.axhline(i, color=inactivity_color, linewidth=10)

        # Add vertical lines to separate duplicate plot
        plt.axvline(bins_per_day, linewidth=2, linestyle="--", color=midline_color)

        # Configure xticks
        num_ticks = 24 * 2 + 1
        ax.set_xticks(np.linspace(0, double_plot_matrix.shape[1], num_ticks))
        ax.set_xticklabels([f"{int(hour % 24)}" for hour in np.linspace(0, 2 * 24, num_ticks)],
                           rotation=0
        )

        # Configure yticks
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Add custom legend for activity and inactivity
        legend_elements = [
            Line2D([0], [0], color=activity_color, lw=4, label="Activity Period"),
            Line2D([0], [0], color=inactivity_color, lw=4, label="Inactivity Period"),
            Line2D([0], [0], color=midline_color, lw=4, label="Double-plot Line"),
        ]
        ax.legend(handles=legend_elements, loc=legend_loc, fontsize=legend_fontsize)

        # Label the plot and axis
        ax.set_title(title)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Day")

        # Display the plot
        if show:
            plt.show()

        # Return figure and axes
        return fig, ax


def main():
    # data = np.array([1, 2, 5, 4, 2, 3, 6, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    # time = np.arange(0, len(data) * 15, 15)
    # data = np.array([
    #   500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
    #  500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500,500,500,0,0,0,0,0,0,0,0,0,500,500,500, 0,0,0,0,0,0,0,0,0,500
    # ])
    # Generate the time vector
    # time = np.arange(0, len(data) * 15, 15)
    # schedule = Light(time_vector=time, light_vector=data)
    # print(schedule.light_vector)
    # mask_1 = schedule.create_mask(120, 360)

    # schedule.interpolate_mask(mask_1)
    # print(schedule.light_vector)
    # schedule = Light.create(total_days=10,light_on_hours=16, bins_per_hour = 10,schedule_starts_at=8,low=500,high=500)
    # Social jet lag schedule
    schedule_base = Light.create(
        total_days=7,
        light_on_hours=16,
        bins_per_hour=10,
        schedule_starts_at=8,
        low=100,
        high=100,
    )
    schedule_end = Light.create(
        total_days=2,
        light_on_hours=3,
        bins_per_hour=10,
        schedule_starts_at=8,
        low=100,
        high=100,
    )
    schedule_add = Light.create(
        total_days=2,
        light_on_hours=3,
        bins_per_hour=10,
        schedule_starts_at=0,
        low=100,
        high=100,
    )
    # a = schedule_base - schedule_end
    schedule = schedule_base.__sub__(schedule_end, shift=5 * 24)
    schedule = schedule.__add__(schedule_add, shift=5 * 24)
    figure, axes = schedule.actogram(show=False)
    # figure.savefig("actogram.png")
    # figure



    # plt.plot(schedule.light_vector)
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Light (lux)")
    # plt.title("Synthetic light schedule")
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
