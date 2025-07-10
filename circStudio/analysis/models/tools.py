from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import re

class Tools:
    """
    Functions related to time manipulations for actigraphy analysis.

    This class provides utilities to handle time-based operations commonly required in actigraphy data processing,
    such as converting between time formats and calculating time differences.

    Methods:
    -------
    difference_hhmm():
        Helper function to calculate the smallest possible difference between two times in HH:MM format.

    difference_decimal():
         Helper function to calculate the smallest possible difference between two times in decimal format (h)
    """

    @classmethod
    def difference_hhmm(cls, start_time:str, end_time:str) -> tuple:
        """
        Calculate the smallest possible difference between two times in HH:MM format.
        :param start_time: The start time formated as 'HH:MM'
        :param end_time: The end time formated as 'HH:MM'
        :return: A tuple containing the smallest time difference in hours, minutes, and the direction of the difference.
                 The direction is indicated by '+' (clockwise) if end_time is after start_time or '-' (anti-clockwise)
                 if the time difference crosses midnight.
        """
        # Define time format and convert input strings to datetime objects
        time_format = '%H:%M'
        start_time = datetime.strptime(start_time, time_format)
        end_time = datetime.strptime(end_time, time_format)

        if end_time >= start_time:
            # Case 1: end_time is after or equal to start_time within the same 24-hour cycle
            diff = end_time - start_time
        else:
            # Case 2: end_time is before start_time, meaning it falls on the next day
            diff = (end_time + timedelta(days=1)) - start_time

        # Calculate the wrap-around difference (i.e., the complementary difference crossing midnight)
        wrap_around_diff = timedelta(days=1) - diff

        # Choose the smallest time difference (either direct or wrap-around)
        smallest_diff = min(diff, wrap_around_diff)

        # Determine the direction of movement: '+' for clockwise, '-' for anti-clockwise
        if diff < wrap_around_diff:
            movement = '+'
        else:
            movement = '-'

        # Convert the time difference into hours and minutes
        hours, remainder = divmod(smallest_diff.seconds, 3600)
        minutes = remainder //60
        return hours, minutes, movement

    @classmethod
    def difference_decimal(cls, start_time:float, end_time:float) -> tuple:
        """
        Calculate the smallest possible difference between two times in decimal format (h).
        :param start_time: The start time formated as decimal (float)
        :param end_time: The end time formated as decimal (float)
        :return: A tuple containing the smallest time difference in decimal format, and the direction of the difference.
                 The direction is indicated by '+' (clockwise) if end_time is after start_time or '-' (anti-clockwise)
                 if the time difference crosses midnight.
        """

        if end_time >= start_time:
            # Case 1: end_time is after or equal to start_time within the same 24-hour cycle
            diff = end_time - start_time
        else:
            # Case 2: end_time is before start_time, meaning it falls on the next day
            diff = (end_time + 24) - start_time

        # Calculate the wrap-around difference (i.e., the complementary difference crossing midnight)
        wrap_around_diff = 24 - diff

        # Choose the smallest time difference (either direct or wrap-around)
        smallest_diff = min(diff, wrap_around_diff)

        # Determine the direction of movement: '+' for clockwise, '-' for anti-clockwise
        if diff < wrap_around_diff:
            movement = '+'
        else:
            movement = '-'

        # Convert the time difference into hours and minutes
        return smallest_diff, movement
    @classmethod
    def _circular_distance(cls,angle_1, angle_2):
        """
        Helper function to calculate the smallest possible difference between two angles in radians.
        :param angle_1: first angle.
        :param angle_2: second angle.
        :return: The smallest circular distance in radians.
        Source: https://insideainews.com/2021/02/12/circular-statistics-in-python-an-intuitive-intro/
        """
        # min((angle_1,angle_2), [2pi - (angle_1, angle_2)]) = pi - |pi - |angle_1-angle_2|
        return np.pi - np.abs(np.pi - np.abs(angle_1 - angle_2))

    @classmethod
    def hhmm_to_h(cls,hhmm: str) -> float:
        """
        Convert a HH:MM string to a decimal hour.
        :param hhmm: time in HH:MM format.
        :return: time converted to decimal hours.
        :raises: ValueError: if the input is not in HH:MM format.
        """
        if re.match(r'^\d{1,2}:\d{2}$', hhmm):
            hours,minutes = map(int, hhmm.split(':'))
            return hours + minutes/60
        else:
            raise ValueError('hhmm must be in HH:MM format')

    @classmethod
    def hours_rads(cls, flag:str, x:float) -> float:
        """
        Convert between hours and radians

        :param flag: Conversion direction. Use 'hours -> rads' to convert hours to radians,
            and 'rads -> hours' to convert radians to hours.
        :param x: The value to convert. Should be in hours if `flag` is 'hours -> rads',
            or in radians if `flag` is 'rads -> hours'.
        :return: The converted value in radians (if converting from hours) or in hours
            (if converting from radians).
        :raises: ValueError: if an invalid flag is provided.
        """
        match flag:
            case r'hours -> rads':
                return (x * 2 * np.pi)/24
            case r'rads -> hours':
                return (12 * x)/np.pi
            case _:
                raise ValueError('Invalid flag. Use \'hours -> rads\' or \'rads -> hours\'.')


class Cir_Descriptive_Stats(Tools):
    """
    Compute descriptive statistics on circular data, extending from the Tools class.

    This class provides methods for calculating common circular statistics such as the resultant length,
    circular mean, and circular median for a given vector of angles.
    """
    @classmethod
    def resultant_length(cls,angle_vector):
        """
        Calculate the resultant vector length for a given set of angles.

        :param angle_vector (np.ndarray): An array of angles in radians.
        :return float: The mean resultant vector length, a measure of concentration
               around the mean direction, between 0 and 1.
        """

        # Total length of the vector
        n = len(angle_vector)

        # Sum of cosines and sines
        cos_sum = np.sum(np.cos(angle_vector))
        sin_sum = np.sum(np.sin(angle_vector))

        # Calculate the resultant vector length and normalize by n
        return np.sqrt(cos_sum**2 + sin_sum**2) / n

    @classmethod
    def circular_variance(cls,angle_vector):
        """
        Calculate the circular variance for a given set of angles.
        :param angle_vector (np.ndarray): An array of angles in radians.
        :return: the circular variance, between 0 and 1. Lower values indicate
        higher concentration around the mean direction.
        """
        # Circular variance is defined as 1 minus the mean resultant length (Fisher,1995)
        return 1 - cls.resultant_length(angle_vector)

    @classmethod
    def circular_std(cls,angle_vector):
        """
        Calculate the circular standard deviation for a given set of angles.
        :param angle_vector (np.ndarray): An array of angles in radians.
        :return float: the circular standard deviation, representing the spread of
        angles around the mean direction.
        """
        # Get the circular variance
        r = cls.resultant_length(angle_vector)

        # Compute the circular standard deviation using the circular variance (Mardia, 1972)
        return np.sqrt(-2 * np.log(r)) if r > 0 else 0

    @classmethod
    def circular_mean(cls, angle_vector):
        """
        Calculate the circular mean of a vector of angles.

        The circular mean is calculated as the arctangent of the sum of sines over the sum of cosines of the angles.

        :param angle_vector: A numpy array or list of angles (in radians).
        :return: Circular mean of the angles (in radians).
        """
        sin_sum = np.sum(np.sin(angle_vector))
        cos_sum = np.sum(np.cos(angle_vector))
        return np.arctan2(sin_sum, cos_sum) % (2 * np.pi)

    @classmethod
    def circular_median(cls,angle_vector):
        """
        Calculate a circular median.
        :param circular_array: a circular data vector.
        :return: a circular median for that circular data vector.
        Source: https://insideainews.com/2021/02/12/circular-statistics-in-python-an-intuitive-intro/
        """
        # For each angle in the array, check if it works as a median (i.e., minimize the distance to all other points)
        # Calculate the distance of a candidate angle to any other angle within the circular array
        dist = [sum([cls._circular_distance(candidate_mid_angle,any_other_angle) for any_other_angle in angle_vector])
                for candidate_mid_angle in angle_vector]

        # If number of elements is even, find the two values in the middle and compute the mean
        if len(angle_vector) % 2 == 0:
            # Order the values
            sorted_dist = np.argsort(dist)
            #Find the two mid_angles and return the mean
            mid_angles = angle_vector[sorted_dist[0:2]]
            return np.mean(mid_angles)
        else:
            # Find the angle that minimizes the total absolute distance to all other points in the data.
            return signal[np.argmin(dist)]

class Cir_Inference_Stats(Cir_Descriptive_Stats):
    """
        A class for performing inferential statistics on circular data, extending Cir_Descriptive_Stats.

        This class inherits descriptive statistics methods and adds functionality for hypothesis testing.
    """
    @classmethod
    def permutation_test(cls, input_data, permutations):
        # Dataframe containing data with the relevant groups in columns
        group_data = input_data

        # Process group data for permutation testing
        groups = []
        for group in group_data:
            new_entry = group_data[group].to_numpy()
            groups.append(new_entry)

        # Compute the observed variances
        observed_variances = []
        for group in groups:
            observed_variances.append(cds.circular_variance(group))

        # Store variances for the null hypothesis scenario
        null_hypothesis_variances = []

        # Construct a pool of all group data
        all_data_vector = np.concatenate(groups)

        # Permutate your data
        n_permutations = permutations
        rng = np.random.default_rng()
        for _ in range(n_permutations):
            # Permutate all the data
            permutated_data = rng.permutation(all_data_vector)

            # Split different groups with the same length as the original
            permutated_groups = []
            start_index = 0
            for group in groups:
                end_index = start_index + len(group)
                permutated_group = permutated_data[start_index:end_index]
                permutated_groups.append(permutated_group)
                start_index = end_index

            # Compute the variances of the null hypothesis scenario
            permutated_variances = [cds.circular_variance(permutated_group) for permutated_group in permutated_groups]
            null_hypothesis_variances.extend(permutated_variances)

        null_hypothesis_variances = np.array(null_hypothesis_variances)

        # Calculate p-values for each observed variance
        p_values = [
            np.mean(null_hypothesis_variances >= observed) for observed in observed_variances
        ]

        # Output results
        return p_values


def main():
    #print(Time.difference_hhmm('10:00', '08:00'))
    #(Time.difference_hhmm('09:27', '00:43'))
    #print(Time.difference_hhmm('22:55', '06:48'))

    #print(Time.difference_decimal(10, 8))
    #print(Time.difference_decimal(9.45, 0.716666667))
    #print(Time.difference_decimal(22.916667, 6.8))
    circular_data = np.random.uniform(0,2*np.pi,10)
    #print(np.sort(circular_data))
    print(Cir_Descriptive_Stats.circular_median(circular_data))
    #print(Tools.hhmm_to_h('7:09'))
    #print(Tools.hours_rads('hours -> rads', 3))


if __name__ == '__main__':
    main()
