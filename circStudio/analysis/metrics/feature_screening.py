from circStudio import *
from circStudio.models.light_tools import *
from circStudio.models.math_models import *
from circStudio.models.tools import *
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from scipy.ndimage import median_filter

class FeatureScreening:
    def __init__(self, data_path: str, mask_path: str, levels: list):
        self.data_path = data_path
        self.mask_path = mask_path
        self.levels = levels

    def panel(self):
        output = []

        for level in self.levels:
            source = os.path.join(self.data_path, level)

            for filename in os.listdir(source):
                fpath = os.path.join(source, filename)
                if os.path.isfile(fpath) and filename.endswith('.txt'):
                    # ----------------------------------
                    # Load & preprocess recording
                    # ----------------------------------
                    raw = io.read_atr(fpath)
                    raw.inactivity_length = None
                    label = os.path.splitext(os.path.basename(fpath))[0]

                    if self.mask_path is not None:
                        raw.add_mask_periods(os.path.join(self.mask_path, f'{label}.csv'))
                        raw.apply_filters(mask=True)

                    # ----------------------------------
                    # Compute actigraphy features
                    # ----------------------------------
                    # Median light exposure level per epoch
                    exp_level = raw.light.light_exposure_level(agg='median').loc['LIGHT']

                    ##Sleep Regularity Index (using the Roenneberg algorithm)
                    sri = raw.SleepRegularityIndex(freq='10min', algo='Roenneberg')

                    # Sleep Midpoint (using the Roenneberg algorithm)
                    sleep_midpoint = raw.SleepMidPoint(freq='10min', to_td=False, algo='Roenneberg')

                    # Median duration of major and minor sleep bouts
                    # 1. Create empty dataframe to store sleep bouts
                    sleep_bouts = pd.DataFrame()

                    # 2. Calculate activity onset and offset
                    activity_onset, activity_offset = raw.Roenneberg_AoT()

                    # 2.1. Assuming sleep_onset = activity_offset & sleep_offset = activity_onset
                    sleep_bouts['date'] = activity_onset.date
                    sleep_bouts['start_time'] = activity_offset
                    sleep_bouts['stop_time'] = activity_onset

                    # 3. Calculate the duration of the sleep/rest episode
                    sleep_bouts['duration'] = sleep_bouts['stop_time'] - sleep_bouts['start_time']

                    # 4. From the list of sleep bouts grouped by date, identify the major sleep bout
                    major_sleep_bouts = sleep_bouts.loc[sleep_bouts.groupby('date')['duration'].idxmax()]

                    # 4.1. Remove major sleep bouts from the sleep bout set to get a set of minor sleep bouts
                    minor_sleep_bouts = sleep_bouts.drop(major_sleep_bouts.index)

                    # 5. Median duration for major sleep bout (minutes)
                    average_duration_major = major_sleep_bouts['duration'].median().total_seconds() / 60

                    # 6. Median duration for minor sleep bouts (minutes)
                    average_duration_minor = minor_sleep_bouts['duration'].median().total_seconds() / 60

                    # Assign 0 if no minor sleep bouts are detected (biologically meaningful: zero duration ≠ missing data)
                    if np.isnan(average_duration_minor):
                        average_duration_minor = 0

                    # Median sleep bout duration across all sleep bouts (minutes)
                    td = pd.Series(raw.sleep_durations(duration_min='10min', algo='Roenneberg'))
                    sleep_duration = td / pd.Timedelta('1min')  # From Timedelta to minutes
                    sleep_bout_median = sleep_duration.median()  # Median

                    # Median activity duration (minutes)
                    td_act = pd.Series(raw.active_durations(duration_min='10min', algo='Roenneberg'))
                    minutes_activity = td_act / pd.Timedelta('1min')
                    activity_median = minutes_activity.median()  # Median

                    # Probability of transitioning from rest to activity after activity offset
                    kra = raw.kRA(4, start='AoffT', freq='10min')

                    # Probability of transitioning from activity to rest after activity onset
                    kar = raw.kAR(4, start='AonT', freq='10min')

                    # Cosinor analysis
                    # 1. Initiate a cosinor object
                    cosinor = Cosinor()

                    # 2. Disable inactivity mask (since Cosinor does not tolerate NaN values)
                    raw.mask_inactivity = False

                    # 3. Set and fix the period to 24-h (1440 minutes for a 60-second sampling rate)
                    cosinor.fit_initial_params['Period'].value = 1440
                    cosinor.fit_initial_params['Period'].vary = False

                    # 4. Store the results from the cosinor fit and extract them
                    results = cosinor.fit(raw.data, verbose=False)
                    mesor = results.params['Mesor'].value
                    amplitude = results.params['Amplitude'].value
                    acrophase = results.params['Acrophase'].value

                    # ----------------------------------
                    # Math models of circadian rhythms
                    # ----------------------------------
                    def impute_nan(df, col_name, datetime_col, mask_address):
                        # Open mask periods as a dataframe
                        df_missing = pd.read_csv(mask_address)

                        # Convert mask start and stop times to datetime
                        df_missing['Start_time'] = pd.to_datetime(df_missing['Start_time'])
                        df_missing['Stop_time'] = pd.to_datetime(df_missing['Stop_time'])

                        for _, row in df_missing.iterrows():
                            mask = (df[datetime_col] >= row['Start_time']) & (df[datetime_col] <= row['Stop_time'])
                            df.loc[mask, col_name] = np.nan

                        # Fill NaN with the mean on the given time (math models do not allow for gaps in the time series)
                        df[col_name] = df[col_name].fillna(
                            df[col_name].groupby(df[datetime_col].dt.time).transform("mean"))

                        # Return the original dataframe with NaN gaps imputed
                        return df

                    def light_pipeline(df, mask_path):
                        # Convert 'DATE/TIME' type from str to datetime
                        df['DATE/TIME'] = pd.to_datetime(df['DATE/TIME'], format="%d/%m/%Y %H:%M:%S")

                        # Linearly interpolate consecutive sequences of zeros between 5 and 45 minutes
                        df = impute_nan(df, col_name='LIGHT', datetime_col='DATE/TIME', mask_address=mask_path)

                        # Set DATE/TIME as the index for resampling
                        df.set_index('DATE/TIME', inplace=True)

                        # Resample to 10-minute intervals, taking the mean
                        df = df.resample('10min').mean()

                        # Reset the index
                        df.reset_index(inplace=True)

                        # Create a new column for the date without time
                        df['DATE'] = df['DATE/TIME'].dt.date

                        # Find the first and last date
                        first_day = df['DATE'].min()
                        last_day = df['DATE'].max()

                        # Exclude rows from the first and last days
                        df = df[(df['DATE'] != first_day) & (df['DATE'] != last_day)]

                        # Calculate hours elapsed from first datetime value
                        filtered_df = df[(df['DATE'] != first_day) & (df['DATE'] != last_day)]
                        filtered_min_time = filtered_df['DATE/TIME'].min()
                        first_value = df['DATE/TIME'].min()
                        df['HOURS'] = (df['DATE/TIME'] - filtered_min_time).dt.total_seconds() / 3600

                        # Smooth light
                        df['LIGHT'] = median_filter(df['LIGHT'], 50)

                        # Reset dataframe index
                        df = df.reset_index()

                        # Return clean df
                        return df

                    def model_pipeline(directory, save_address):
                        def compute_models(fpath, mask_path):
                            def create_light(dataframe, mask_path):
                                # Create a dataframe with 'DATE/TIME' and 'LIGHT' columns
                                df_light = dataframe[['DATE/TIME', 'LIGHT']].copy()

                                # Apply the light processing pipeline
                                treated_df = light_pipeline(df_light, mask_path)

                                # Retrieve the first two days of actigraphy data to calculate initial conditions
                                first_two_days = treated_df['DATE/TIME'].dt.date.drop_duplicates()[:2]

                                # Filter rows from the first two days
                                ics_df = treated_df[treated_df['DATE/TIME'].dt.date.isin(first_two_days)]

                                # Create a return the light schedule
                                schedule = Light(time_vector=treated_df.HOURS, light_vector=treated_df.LIGHT)
                                ics_schedule = Light(time_vector=ics_df.HOURS, light_vector=ics_df.LIGHT)
                                return schedule, ics_schedule, treated_df

                            # Forger and Jewett models
                            def cbtmin_vdp(time, x):
                                # Calculate time step (dt) between consecutive time points
                                dt = np.diff(time)[0]

                                # Invert cos(x) to turn the minima into maxima (peaks)
                                inverted_x = -1.0 * x

                                # Identify the indices where minima occur
                                cbtmin_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))

                                # Use the previous indices to find the cbtmin times
                                cbtmin_times = time[cbtmin_indices]

                                # To convert to clock time -> cbtmin_times % 24
                                return cbtmin_times

                            # HannaySP and HannayTP models
                            def cbtmin_hannay(time, phase):
                                # Calculate time step (dt) between consecutive time points
                                dt = np.diff(time)[0]

                                # Invert cos(x) to turn the minima into maxima (peaks)
                                inverted = -1.0 * np.cos(phase)

                                # Identify the indices where minima occur
                                cbtmin_indices, _ = find_peaks(inverted, distance=np.ceil(13.0 / dt))

                                # Use the previous indices to find the cbtmin times
                                cbtmin_times = time[cbtmin_indices]

                                # To convert to clock time -> cbtmin_times % 24
                                return cbtmin_times

                            # Compute dlmos (both VDP and Hannay models)
                            def dlmos(cbt_vector, cbt_to_dlmo=7):
                                return (cbt_vector - cbt_to_dlmo) % 24

                            def create_model(time, light, time_ics, light_ics):
                                # Model class list
                                models = [Forger, Jewett, HannaySP, HannayTP]

                                # Create instances of each model and populate model_instances
                                for model in models:
                                    # Create model to compute initial conditions
                                    ics_model = model(time=time_ics, inputs=light_ics)
                                    ics_model.initial_condition = ics_model.get_initial_conditions(
                                        time_vector=time_ics,
                                        light_vector=light_ics,
                                        loop_number=50
                                    )

                                    # Create model to compute the trajectory based on initial conditions
                                    model_obj = model(time=time, inputs=light)
                                    model_obj.model_states = model_obj.integrate(
                                        time_vector=time,
                                        light_vector=light,
                                        initial_condition=ics_model.initial_condition
                                    )

                                    # Store the result in model collector dictionary
                                    model_name = model.__name__.lower()
                                    vdp_models = ['forger', 'jewett']
                                    if model_name in vdp_models:
                                        dlmo(cbtmin_vdp(time, model_obj.model_states[:, 0]))
                                    else:
                                        dlmo(cbtmin_hannay(time, model_obj.model_states[:, 1]))

                                # Return the created model instances
                                return model_objs

                            # Create light schedule
                            schedule, ics_schedule, data = create_light(raw.df, mask_path)

                            # Create models
                            models = create_model(time=np.asarray(schedule.time_vector),
                                                  light=np.asarray(schedule.light_vector),
                                                  time_ics=np.asarray(ics_schedule.time_vector),
                                                  light_ics=np.asarray(ics_schedule.light_vector)
                                                  )

                            # Collect model outputs in a dictionary
                            model_outputs = {
                                "datetime": np.asarray(data['DATE/TIME']),
                                "hours": schedule.time_vector,
                                "light": schedule.light_vector
                            }

                            # Define model and model variable names
                            model_var_dict = {
                                'forger': ['x_forger', 'xc_forger', 'n_forger'],
                                'jewett': ['x_jewett', 'xc_jewett', 'n_jewett'],
                                'hannaysp': ['r_sp', 'phi_sp', 'n_sp'],
                                'hannaytp': ['r1_tp', 'r2_tp', 'phi1_tp', 'phi2_tp', 'n_tp']
                            }

                            # Extract outputs from each model
                            for model_name, output_vars in model_var_dict.items():
                                model = models[model_name]
                                for i, var_name in enumerate(output_vars):
                                    model_outputs[var_name] = model.model_states[:, i]

                            # Return DataFrame with collected outputs
                            return pd.DataFrame(model_outputs)

                    # Forger and Jewett models
                    def cbtmin_vdp(time, x):
                        # Calculate time step (dt) between consecutive time points
                        dt = np.diff(time)[0]

                        # Invert cos(x) to turn the minima into maxima (peaks)
                        inverted_x = -1.0 * x

                        # Identify the indices where minima occur
                        cbtmin_indices, _ = find_peaks(inverted_x, distance=np.ceil(13.0 / dt))

                        # Use the previous indices to find the cbtmin times
                        cbtmin_times = time[cbtmin_indices]

                        # To convert to clock time -> cbtmin_times % 24
                        return cbtmin_times

                    # HannaySP and HannayTP models
                    def cbtmin_hannay(time, phase):
                        # Calculate time step (dt) between consecutive time points
                        dt = np.diff(time)[0]

                        # Invert cos(x) to turn the minima into maxima (peaks)
                        inverted = -1.0 * np.cos(phase)

                        # Identify the indices where minima occur
                        cbtmin_indices, _ = find_peaks(inverted, distance=np.ceil(13.0 / dt))

                        # Use the previous indices to find the cbtmin times
                        cbtmin_times = time[cbtmin_indices]

                        # To convert to clock time -> cbtmin_times % 24
                        return cbtmin_times

                    # Compute dlmos (both VDP and Hannay models)
                    def dlmos(cbt_vector, cbt_to_dlmo=7):
                        return (cbt_vector - cbt_to_dlmo) % 24

                    for shift in shifts:
                        for i, (start, end) in enumerate(zip(start_dates, end_dates), start=1):
                            reg_dict = {}

                            for model in models:
                                # Filter input dataframe for a specific model, shift, and period
                                df_filtered = df_input[(df_input['model'] == model) & (df_input['shift'] == shift) & (
                                    df_input['date'].between(start, end))].dropna()
                                df_filtered['period'] = f'p{i}'

                                # Define t0 (zero hours elapsed since the beginning)
                                first_day = df_filtered['date'].min().normalize()

                                # Convert dates to hours since first day
                                df_filtered['t'] = (df_filtered['date'] - first_day).dt.total_seconds() / 3600

                                # Group by sb_id
                                for sb, df_sb in df_filtered.groupby('sb_id'):
                                    # Perform linear regression
                                    res = stats.linregress(df_sb['t'], df_sb['cbtmin'] % 24)

                                    # Store results in dictionary
                                    if (sb, shift, f'p{i}') not in reg_dict:
                                        reg_dict[(sb, shift, f'p{i}')] = {'sb_id': sb.lower(), 'shift': shift,
                                                                          'period': f'p{i}'}

                                    reg_dict[(sb, shift, f'p{i}')].update({
                                        f'{model}_r': res.rvalue,
                                        f'{model}_slope': res.slope
                                    })


                    ### Collect entry and add it to the dataframe
                    sb_row = {
                        'ID': label,
                        'Level': level,
                        'ADAT': adat(data=raw.activity),
                        'IS': interdaily_stability(data=raw.activity.resample('1h')),
                        'IV': intradaily_variability(data=raw.activity.resample('1h')),
                         'L5': l5(data=raw.activity)[1],
                         'M10': m10(data=raw.activity)[1],
                         'RA': ra(data=raw.activity),
                         'IS_light': interdaily_stability(data=raw.light),
                         'IV_light': intradaily_variability(data=raw.light),
                         'L5_light': l5(data=raw.activity)[1],
                         'M10_light': m10(data=raw.activity)[1],
                         'L5_onset_light': l5(data=raw.light)[0].total_seconds() / 60,
                         'M10_onset_light': m10(data=raw.light)[0].total_seconds() / 60,
                         'Mlit_10lux': mlit(light=raw.light, threshold=10),
                         'Mlit_100lux': mlit(light=raw.light, threshold=100),
                         'Mlit_500lux': mlit(light=raw.light, threshold=500),
                         'Tat_10lux': TATp(threshold=10, oformat='minute').median(),
                         'Tat_100lux': TATp(threshold=100, oformat='minute').median(),
                         'Tat_100lux': TATp(threshold=500, oformat='minute').median(),
                         'Vat_10lux': VAT(data=raw.light, threshold=10).median(),
                         'Vat_100lux': VAT(data=raw.light, threshold=100).median(),
                         'Vat_500lux': VAT(data=raw.light, threshold=500).median(),
                         'Sri': SleepRegularityIndex(raw.activity.resample('10min'), algo='Roenneberg'),
                         }
                    output.apend(sub_row)

                    # Save intermediate dataframe with actigraphy-derived features computed using pyActigraphy
                    df.to_csv(os.path.join('..', 'data', 'processed', 'raw_act_features.csv'), index=False)
