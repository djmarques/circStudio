from circStudio import *
import pandas as pd
import numpy as np
import os
import scipy.stats as stats

class FeatureScreening:
    def __init__(self, data_path, mask_path, levels):
        self.data_path = data_path
        self.mask_path = mask_path
        self.levels = levels

    def panel(self):
        # Create a list to collect output dictionaries containing variables from each submariner
        output = []

        for level in self.levels:
            source = os.path.join(self.data_path, level)

            for filename in os.listdir(source):
                fpath = os.path.join(source, filename)
                if os.path.isfile(fpath) and filename.endswith('.txt'):
                    # Read actigraphy file from a given submariner
                    raw = io.read_atr(fpath)

                    # Reset any pre-existing inactivity mask
                    raw.inactivity_length = None  # Reset any pre-existing mask

                    # Populate inactivity mask with periods of spurious inactivity determined by visual inspection
                    mask_name = os.path.splitext(os.path.basename(fpath))[0]
                    raw.add_mask_periods(
                        os.path.join('..', 'data', 'processed', 'missings', level, 'csv', f'{mask_name}.csv'))
                    raw.apply_filters(mask=True)  # Apply the mask

                    # Retrieve the current submariner’s shift from a shift lookup table
                    identificator = pd.read_csv(os.path.join('..', 'data', 'raw', 'database', 'sb_shifts.csv'),
                                                index_col='sb_id')
                    shift = identificator.loc[mask_name]['shift']

                    ### Median light exposure level per epoch
                    exp_level = raw.light.light_exposure_level(agg='median').loc['LIGHT']

                    # Sleep-related features
                    ## Sleep Regularity Index (using the Roenneberg algorithm)
                    sri = raw.SleepRegularityIndex(freq='10min', algo='Roenneberg')

                    ## Sleep Midpoint (using the Roenneberg algorithm)
                    sleep_midpoint = raw.SleepMidPoint(freq='10min', to_td=False, algo='Roenneberg')

                    ## Median duration of major and minor sleep bouts
                    ### 1) Create empty dataframe to store sleep bouts
                    sleep_bouts = pd.DataFrame()

                    ### 2) Calculate activity onset and offset
                    activity_onset, activity_offset = raw.Roenneberg_AoT()

                    #### 2.1) Assuming sleep_onset = activity_offset & sleep_offset = activity_onset
                    sleep_bouts['date'] = activity_onset.date
                    sleep_bouts['start_time'] = activity_offset
                    sleep_bouts['stop_time'] = activity_onset

                    ### 3) Calculate the duration of the sleep/rest episode
                    sleep_bouts['duration'] = sleep_bouts['stop_time'] - sleep_bouts['start_time']

                    ### 4) From the list of sleep bouts grouped by date, identify the major sleep bout
                    major_sleep_bouts = sleep_bouts.loc[sleep_bouts.groupby('date')['duration'].idxmax()]

                    #### 4.1) Remove major sleep bouts from the sleep bout set to get a set of minor sleep bouts
                    minor_sleep_bouts = sleep_bouts.drop(major_sleep_bouts.index)

                    ### 5) Median duration for major sleep bout (minutes)
                    average_duration_major = major_sleep_bouts['duration'].median().total_seconds() / 60

                    ### 6) Median duration for minor sleep bouts (minutes)
                    average_duration_minor = minor_sleep_bouts['duration'].median().total_seconds() / 60

                    # Assign 0 if no minor sleep bouts are detected (biologically meaningful: zero duration ≠ missing data)
                    if np.isnan(average_duration_minor):
                        average_duration_minor = 0

                    ## Median sleep bout duration across all sleep bouts (minutes)
                    td = pd.Series(raw.sleep_durations(duration_min='10min', algo='Roenneberg'))
                    sleep_duration = td / pd.Timedelta('1min')  # From Timedelta to minutes
                    sleep_bout_median = sleep_duration.median()  # Median

                    ## Median activity duration (minutes)
                    td_act = pd.Series(raw.active_durations(duration_min='10min', algo='Roenneberg'))
                    minutes_activity = td_act / pd.Timedelta('1min')
                    activity_median = minutes_activity.median()  # Median

                    ## Probability of transitioning from rest to activity after activity offset
                    kra = raw.kRA(4, start='AoffT', freq='10min')

                    ## Probability of transitioning from activity to rest after activity onset
                    kar = raw.kAR(4, start='AonT', freq='10min')

                    ## Cosinor analysis
                    ### 1) Initiate a cosinor object
                    cosinor = Cosinor()

                    ### 2) Disable inactivity mask (since Cosinor does not tolerate NaN values)
                    raw.mask_inactivity = False

                    ### 3) Set and fix the period to 24-h (1440 minutes for a 60-second sampling rate)
                    cosinor.fit_initial_params['Period'].value = 1440
                    cosinor.fit_initial_params['Period'].vary = False

                    ### 4) Store the results from the cosinor fit and extract them
                    results = cosinor.fit(raw.data, verbose=False)
                    mesor = results.params['Mesor'].value
                    amplitude = results.params['Amplitude'].value
                    acrophase = results.params['Acrophase'].value

                    ### Collect entry and add it to the dataframe
                    sb_row = {
                        'sb_id': mask_name,
                        'period': period,
                        'shift': shift,
                        'ADAT': adat(data=raw.activity),
                        'IS': interdaily_stability(data=raw.activity.resample('1h')),
                        'IV': intradaily_variability(data=raw.activity.resample('1h')),
                         'L5': l5(data=raw.activity)[1],
                         'M10': m10(data=raw.activity)[1],
                         'RA': relative_amplitude(data=raw.activity),
                         'IS_light': interdaily_stability(data=raw.light),
                         'IV_light': intradaily_variability(data=raw.light),
                         'L5_light': l5(data=raw.activity)[1],
                         'M10_light': m10(data=raw.activity)[1],
                         'L5_onset_light': l5(data=raw.light)[0].total_seconds() / 60,
                         'M10_onset_light': m10(data=raw.light)[0].total_seconds() / 60,
                         'mlit_10lux': mean_light_timing(light=raw.light, threshold=10),
                         'mlit_100lux': mean_light_timing(light=raw.light, threshold=100),
                         'mlit_500lux': mean_light_timing(light=raw.light, threshold=500),
                         'tat_10lux': time_above_threshold_by_period(threshold=10, oformat='minute').median(),
                         'tat_100lux': time_above_threshold_by_period(threshold=100, oformat='minute').median(),
                         'tat_100lux': time_above_threshold_by_period(threshold=500, oformat='minute').median(),
                         'vat_10lux': values_above_threshold(data=raw.light, threshold=10).median(),
                         'vat_100lux': values_above_threshold(data=raw.light, threshold=100).median(),
                         'vat_500lux': values_above_threshold(data=raw.light, threshold=500).median(),
                         'sri': SleepRegularityIndex(raw.activity.resample('10min'), algo='Roenneberg'),
                         }
                    output.apend(sub_row)

                    # Save intermediate dataframe with actigraphy-derived features computed using pyActigraphy
                    df.to_csv(os.path.join('..', 'data', 'processed', 'raw_act_features.csv'), index=False)
