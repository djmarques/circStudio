import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


class Light:
    def __init__(self, time_vector, light_vector):
        """
        Initialize a Light object with time and light intensity data.
        :param time_vector (numpy array): Arrau of time points
        :param light_vector: Corresponding light intensity values (in lux)
        """
        self.light_vector = light_vector
        self.time_vector = time_vector

    def create_mask(self, min_length_minutes, max_length_minutes):
        # Sampling interval calculation
        sampling_interval = int(self.time_vector[1] - self.time_vector[0])

        # Convert minutes to number of indices based on sampling interval
        min_length_indices = int(min_length_minutes / sampling_interval)
        max_length_indices = int(max_length_minutes / sampling_interval)

        # Identify positions where the light vector is zero
        is_zero = self.light_vector == 0

        # Detect changes from zero to non-zero and vice-versa
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
            self.light_vector[start + 1:end + 2] = interpolated_values

    def downsample(self, factor):
        """
        Adjust the frequency at which light intensity values are sampled. The original
        data is sampled at a specific interval (e.g., 1 minute, 1 hour). When you provide
        a downsampling interval, this method increases the time between each sampled point
        by a factor of the given interval.
        :param factor (float): downsampling factor
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
    ):
        """
        Create a synthetic light schedule.

        :param total_days (int): Number of days for the synthetic light schedule
        :param light_on_hours (float): Number of hours per day with lights on
        :param bins_per_hour (int): Time resolution in bins per hour
        :param schedule_starts_at (float): Starting hour of the synthetic light schedule
        :param low: Minimum light intensity during the light-on period (in lux)
        :param high: Maximum light intensity during the light-on period (in lux)

        :return Light: a new Light object containing the generated schedule
        """
        # Calculate the number of bins for light-on and light-off periods
        light_on_bins = int(light_on_hours * bins_per_hour)
        light_off_bins = int((24 - light_on_hours) * bins_per_hour)

        # Instantiate a new generator
        #rng = np.random.default_rng()

        # Adjust light intensity bounds if they are identical
        if low == high:
            #high += 1
            # Instantiate a new generator
            rng = np.random.default_rng()

            # Generate random light intensity values for the light-on period
            light_on_variation = np.full(shape=(1,light_on_bins), fill_value=low)[0]

            # Create a daily schedule with light-on and light-off periods
            daily_schedule = np.concatenate([light_on_variation, np.zeros(light_off_bins)])

            # Determine the start position for the light-on period
            start_bin = schedule_starts_at * bins_per_hour

            # Shift the schedule to the specified start time
            shifted_schedule = np.roll(daily_schedule, start_bin)

            # Repeat the daily schedule for the total number of days
            light_vector = np.tile(shifted_schedule, total_days)

            # Calculate the time step and generate the time vector
            dt = 1 / bins_per_hour
            time_vector = np.arange(0, len(light_vector) * dt, dt)

            # Return the new Light instance
            return cls(time_vector, light_vector)

        else:
            # Instantiate a new generator
            rng = np.random.default_rng()

            # Generate random light intensity values for the light-on period
            light_on_variation = rng.uniform(low=low, high=high, size=light_on_bins)

            # Create a daily schedule with light-on and light-off periods
            daily_schedule = np.concatenate([light_on_variation, np.zeros(light_off_bins)])

            # Determine the start position for the light-on period
            start_bin = schedule_starts_at * bins_per_hour

            # Shift the schedule to the specified start time
            shifted_schedule = np.roll(daily_schedule, start_bin)

            # Repeat the daily schedule for the total number of days
            light_vector = np.tile(shifted_schedule, total_days)

            # Calculate the time step and generate the time vector
            dt = 1 / bins_per_hour
            time_vector = np.arange(0, len(light_vector) * dt, dt)

            # Return the new Light instance
            return cls(time_vector, light_vector)

    def __add__(self, other, shift=0):
        """
        Add two light schedules, optionally with a time shift.
        :param other (Light): Another Light instance for addition
        :param shift (float): Time shift for the second schedule being added
        :return Light: A new Light instance representing the combined schedule
        """
        # Calculate bins per hour for the current schedule
        bins_per_hour = int(1 / (self.time_vector[1] - self.time_vector[0]))

        # Convert the time shift from hours to bins
        shift *= bins_per_hour

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
        Sibtract another light schedule, optionally with a time shift

        :param other (Light): Another Light instance to subtract
        :param shift (float): Time shift for the other schedule
        :return Light: A new Light instance representing the resulting schedule
        """
        # Calculate bins per hour for the current schedule
        bins_per_hour = int(1 / (self.time_vector[1] - self.time_vector[0]))

        # Convert the time shift from hours to bins
        shift *= bins_per_hour

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
        Scale the light schedule by a scalar factor.
        :param scalar (float): Scaling factor
        :return Light: A new Light instance with the scaled light intensities
        """
        return Light(self.time_vector, self.light_vector * scalar)

    def __truediv__(self, scalar):
        """
        Divide the light schedule by a scalar factor

        :param scalar (float): Dividing factor
        :return Light: A new Light instance with the scaled light intensities
        """
        return Light(self.time_vector, self.light_vector / scalar)

    def __str__(self):
        """
        Obtain a string containing the light light intensity for each point in time
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

    def actogram_binary(self,
                        size=(12,6),
                        title="Actogram",
                        legend_loc="upper left",
                        legend_fontsize=10,
                        activity_color="#404040",
                        inactivity_color="#DCDCDC",
                        midline_color="red"):
        # Determine the number of bins per day
        bins_per_day = int(24 / np.diff(self.time_vector)[0])

        # Determine the total number of days
        total_time = self.time_vector[-1]  - self.time_vector[0]
        num_days = int(np.ceil(total_time / 24))

        # Reshape the light vector for the actogram
        light_matrix = self.light_vector.reshape(num_days,bins_per_day)

        # Create double-plotted data
        double_plot_matrix = np.tile(light_matrix, 2)

        # Create figure to store the heatmap
        plt.figure(figsize=size)

        # Create a custom color map
        custom_cmap = LinearSegmentedColormap.from_list("pastel", [inactivity_color,activity_color])

        # Create the heatmap
        sns.heatmap(double_plot_matrix,cmap = custom_cmap, cbar=False, yticklabels=np.arange(1, light_matrix.shape[0] + 1))

        # Add horizontal lines to separate rows (days)
        for i in range(0, double_plot_matrix.shape[0]+1):
            plt.axhline(i, color=inactivity_color, linewidth=10)

        # Add vertical lines to separate duplicate plot
        plt.axvline(bins_per_day, linewidth=2, linestyle='--', color=midline_color)

        # Configure xticks
        num_ticks = 24 * 2 + 1
        plt.xticks(
            rotation=0,
            ticks=np.linspace(0, double_plot_matrix.shape[1], num_ticks),
            labels=[f"{int(hour % 24)}" for hour in np.linspace(0, 2 * 24, num_ticks)]
        )

        # Configure yticks
        plt.yticks(rotation=0)

        # Add custom legend for activity and inactivity
        legend_elements = [
            Line2D([0], [0], color=activity_color, lw=4, label='Activity Period'),
            Line2D([0], [0], color=inactivity_color, lw=4, label='Inactivity Period'),
            Line2D([0], [0], color=midline_color, lw=4, label='Double-plot Line')
        ]
        plt.legend(handles=legend_elements, loc=legend_loc, fontsize=legend_fontsize)

        # Label the plot and axis
        plt.title(title)
        plt.xlabel("Time (hours)")
        plt.ylabel("Day")
        plt.show()


def main():
    #data = np.array([1, 2, 5, 4, 2, 3, 6, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    #time = np.arange(0, len(data) * 15, 15)
    #data = np.array([
     #   500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
      #  500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500,500,500,0,0,0,0,0,0,0,0,0,500,500,500, 0,0,0,0,0,0,0,0,0,500
    #])

    # Generate the time vector
    #time = np.arange(0, len(data) * 15, 15)
    #schedule = Light(time_vector=time, light_vector=data)
    #print(schedule.light_vector)
    #mask_1 = schedule.create_mask(120, 360)

    #schedule.interpolate_mask(mask_1)
    #print(schedule.light_vector)
    #schedule = Light.create(total_days=10,light_on_hours=16, bins_per_hour = 10,schedule_starts_at=8,low=500,high=500)
    # Social jet lag schedule
    schedule_base = Light.create(total_days=7, light_on_hours=16, bins_per_hour=10, schedule_starts_at=8, low=100,
                                 high=100)
    schedule_end = Light.create(total_days=2, light_on_hours=3, bins_per_hour=10, schedule_starts_at=8, low=100,
                                high=100)
    schedule_add = Light.create(total_days=2, light_on_hours=3, bins_per_hour=10, schedule_starts_at=0, low=100,
                                high=100)
    # a = schedule_base - schedule_end
    schedule = schedule_base.__sub__(schedule_end, shift=5 * 24)
    schedule = schedule.__add__(schedule_add, shift=5*24)
    schedule.actogram_binary()


    #plt.plot(schedule.light_vector)
    #plt.xlabel("Time (hours)")
    #plt.ylabel("Light (lux)")
    #plt.title("Synthetic light schedule")
    #plt.grid()
    #plt.show()


if __name__ == "__main__":
    main()
