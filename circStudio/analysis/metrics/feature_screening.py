from circStudio.analysis.metrics import *
from circStudio.analysis.sleep import *
import circStudio
from ..models.light_tools import *
from ..models.math_models import *
from ..models.tools import *
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from scipy.ndimage import median_filter
import pingouin as pg
from itertools import combinations
from sklearn.metrics import roc_curve, auc


class FeatureScreening:
    def __init__(self,
                 data_path: str,
                 datetime_column: str,
                 light_column: str,
                 factor: str,
                 levels: list = None,
                 mask_path=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.factor = factor
        self.levels = levels
        self.datetime = datetime_column
        self.light = light_column
        self.panel = None

    def calculate_panel(self) -> pd.DataFrame:
        """
        Calculate panel of actigraphy-derived features.
        """
        panel = []

        for level in self.levels:
            source = os.path.join(self.data_path, level)

            for filename in os.listdir(source):
                fpath = os.path.join(source, filename)
                if os.path.isfile(fpath) and filename.endswith('.txt'):
                    # ----------------------------------
                    # Load & preprocess recording
                    # ----------------------------------
                    raw = circStudio.io.read_atr(fpath)
                    raw.inactivity_length = None
                    label = os.path.splitext(os.path.basename(fpath))[0]

                    if self.mask_path is not None:
                        raw.add_mask_periods(os.path.join(self.mask_path, f'{label}.csv'))
                        raw.apply_filters(mask=True)

                    # ----------------------------------
                    # Cosinor analysis
                    # ----------------------------------
                    # 1. Initiate a cosinor object
                    cosinor = circStudio.Cosinor()

                    # 2. Disable inactivity mask (since Cosinor does not tolerate NaN values)
                    raw.mask_inactivity = False

                    # 3. Set and fix the period to 24-h (1440 minutes for a 60-second sampling rate)
                    cosinor.fit_initial_params['Period'].value = 1440
                    cosinor.fit_initial_params['Period'].vary = False

                    # 4. Store the results from the cosinor fit and extract them
                    results = cosinor.fit(raw.activity, verbose=False)
                    mesor = results.params['Mesor'].value
                    amplitude = results.params['Amplitude'].value
                    acrophase = results.params['Acrophase'].value

                    # ----------------------------------
                    # Math models of circadian rhythms
                    # ----------------------------------
                    def impute_nan(col_name, datetime_col):
                        # Open mask periods as a dataframe
                        df_missing = pd.read_csv(self.mask_path)

                        # Convert mask start and stop times to datetime
                        df_missing['Start_time'] = pd.to_datetime(df_missing['Start_time'])
                        df_missing['Stop_time'] = pd.to_datetime(df_missing['Stop_time'])

                        for _, row in df_missing.iterrows():
                            mask = (raw.df[datetime_col] >= row['Start_time']) & (raw.df[datetime_col] <= row['Stop_time'])
                            raw.df.loc[mask, col_name] = np.nan

                        # Fill NaN with the mean on the given time (math models do not allow for gaps in the time series)
                        raw.df[col_name] = raw.df[col_name].fillna(
                            raw.df[col_name].groupby(raw.df[datetime_col].dt.time).transform("mean"))

                        # Return the original dataframe with NaN gaps imputed
                        return raw.df

                    def light_pipeline(df):
                        # Convert 'DATE/TIME' type from str to datetime
                        df[self.datetime] = pd.to_datetime(df[self.datetime], format="%d/%m/%Y %H:%M:%S")

                        # Linearly interpolate consecutive sequences of zeros between 5 and 45 minutes
                        if self.mask_path is not None:
                            df = impute_nan(col_name=self.light, datetime_col=self.datetime)

                        # Set DATE/TIME as the index for resampling
                        df.set_index(self.datetime, inplace=True)

                        # Resample to 10-minute intervals, taking the mean
                        df = df.resample('10min').mean()

                        # Reset the index
                        df.reset_index(inplace=True)

                        # Create a new column for the date without time
                        df['DATE'] = df[self.datetime].dt.date

                        # Find the first and last date
                        first_day = df['DATE'].min()
                        last_day = df['DATE'].max()

                        # Exclude rows from the first and last days
                        df = df[(df['DATE'] != first_day) & (df['DATE'] != last_day)]

                        # Calculate hours elapsed from first datetime value
                        filtered_df = df[(df['DATE'] != first_day) & (df['DATE'] != last_day)]
                        filtered_min_time = filtered_df[self.datetime].min()
                        first_value = df[self.datetime].min()
                        df['HOURS'] = (df[self.datetime] - filtered_min_time).dt.total_seconds() / 3600

                        # Smooth light
                        df[self.light] = median_filter(df[self.light], 50)

                        # Reset dataframe index
                        df = df.reset_index()

                        # Return clean df
                        return df

                    def create_light():
                        # Create a dataframe with 'DATE/TIME' and 'LIGHT' columns
                        df_light = raw.df.reset_index()[[self.datetime, self.light]].copy()

                        # Apply the light processing pipeline
                        treated_df = light_pipeline(df_light)

                        # Retrieve the first two days of actigraphy data to calculate initial conditions
                        first_two_days = treated_df[self.datetime].dt.date.drop_duplicates()[:2]

                        # Filter rows from the first two days
                        ics_df = treated_df[treated_df[self.datetime].dt.date.isin(first_two_days)]

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
                        model_output = {}

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

                            dlmos_ts = None
                            # Calculate time series of DLMO values
                            if model_name in vdp_models:
                                dlmos_ts = dlmos(cbtmin_vdp(time, model_obj.model_states[:, 0])) % 24
                            else:
                                dlmos_ts = dlmos(cbtmin_hannay(time, model_obj.model_states[:, 1])) % 24

                            # Process and perform linear regression
                            dlmos_unwrapped = np.unwrap((dlmos_ts / 24) * 2 * np.pi)
                            days = np.arange(len(dlmos_ts))
                            line = stats.linregress(days, dlmos_unwrapped)
                            slope = (line.slope * 24)/(2 * np.pi)
                            rvalue = line.rvalue

                            # Save result
                            model_output[f'{model_name}_slope'] = slope
                            model_output[f'{model_name}_rvalue'] = rvalue

                        # Return the created model instances
                        return model_output

                    # Create light schedule
                    schedule, ics_schedule, data = create_light()

                    # Create models
                    model_results = create_model(time=np.asarray(schedule.time_vector),
                                          light=np.asarray(schedule.light_vector),
                                          time_ics=np.asarray(ics_schedule.time_vector),
                                          light_ics=np.asarray(ics_schedule.light_vector)
                                          )

                    # Collect entry and add it to the dataframe
                    sb_row = {
                        'ID': label,
                        'Level': level,
                        'ADAT': adat(data=raw.activity),
                        'IS': IS(data=raw.activity.resample('1h').mean()),
                        'IV': IV(data=raw.activity.resample('1h').mean()),
                        'L5': l5(data=raw.activity)[1],
                        'M10': m10(data=raw.activity)[1],
                        'RA': ra(data=raw.activity),
                        'IS_light': IS(data=raw.light),
                        'IV_light': IV(data=raw.light),
                        'L5_light': l5(data=raw.activity)[1],
                        'M10_light': m10(data=raw.activity)[1],
                        'L5_onset_light': l5(data=raw.light)[0].total_seconds() / 60,
                        'M10_onset_light': m10(data=raw.light)[0].total_seconds() / 60,
                        'Mlit_10lux': mlit(light=raw.light, threshold=10),
                        'Mlit_100lux': mlit(light=raw.light, threshold=100),
                        'Mlit_500lux': mlit(light=raw.light, threshold=500),
                        'Tat_10lux': TATp(data=raw.light, threshold=10, oformat='minute').median(),
                        'Tat_100lux': TATp(data=raw.light, threshold=100, oformat='minute').median(),
                        'Tat_500lux': TATp(data=raw.light, threshold=500, oformat='minute').median(),
                        'Vat_10lux': VAT(data=raw.light, threshold=10).median(),
                        'Vat_100lux': VAT(data=raw.light, threshold=100).median(),
                        'Vat_500lux': VAT(data=raw.light, threshold=500).median(),
                        'kRA': kRA(data=raw.activity),
                        'kAR': kAR(data=raw.activity),
                        'Sri': SleepRegularityIndex(raw.activity.resample('10min').mean(), algo='Roenneberg'),
                        'Sleep_midpoint': SleepMidPoint(data=raw.activity.resample('10min').mean(),
                                                        to_td=False,
                                                        algo='Roenneberg'),
                        'Mesor': mesor,
                        'Amplitude': amplitude,
                        'Acrophase': acrophase,
                        'Exp_level': light_exposure(light=raw.light, agg='median'),
                        'Forger_slope': model_results['forger_slope'],
                        'Jewett_slope': model_results['jewett_slope'],
                        'Hannaysp_slope': model_results['hannaysp_slope'],
                        'Hannaytp_slope': model_results['hannaytp_slope'],
                        'Forger_rvalue': model_results['forger_rvalue'],
                        'Jewett_rvalue': model_results['jewett_rvalue'],
                        'Hannaysp_rvalue': model_results['hannaysp_rvalue'],
                        'Hannaytp_rvalue': model_results['hannaytp_rvalue'],
                    }
                    panel.append(sb_row)
        self.panel = pd.DataFrame(panel)
        return self.panel

    def permutation_test(self, n_permutations=1000, repeated_measures=False, block=None):
        if repeated_measures:
            if block is None:
                raise ValueError("Block must be provided for repeated measures permutation.")

        def cohens_f(df, feature):
            eta_sqr = pg.anova(dv=feature,
                                  between='Level',
                                  data=df,
                                  detailed=False,
                                  effsize='n2')['n2'][0]
            return np.sqrt(eta_sqr / (1 - eta_sqr))

        def process_feature(feature, n_permutations):
            f_observed = cohens_f(df=self.panel, feature=feature)

            # Create list to store null-distribution F for periods and shifts
            null_distribution = []

            if not repeated_measures:
                # Permutation loop
                for _ in range(n_permutations):
                    # Permutate levels
                    df_permutation = self.panel.copy()
                    df_permutation['Level'] = df_permutation['Level'].sample(frac=1).values

                    # Compute F statistics (for shift)
                    f_null = cohens_f(df_permutation, feature=feature)
                    null_distribution.append(f_null)
                return f_observed, null_distribution
            else:
                # Permutation loop
                for _ in range(n_permutations):
                    # Permutate levels (repeated measure)
                    df_permutation = self.panel.copy()
                    perm = df_permutation.groupby(block)['Label'].transform(lambda x: x.sample(frac=1).values)
                    df_permutation['Label'] = perm

                    # Compute F statistics for period
                    f_null = cohens_f(df_permutation, feature=feature)
                    null_distribution.append(f_null)
                return f_observed, null_distribution

        def fdr(null_distribution, f_observed, n_permutations):
            """
            null_dist (dataframe containing a null distribution of f values generated by permutations) - dataframe
            obs_stat (dataframe containing the observed f value) - dataframe
            n_permutations (number of permutations previously used) - int
            """
            if np.isnan(f_observed):
                return np.nan
            tol = max(1e-14, abs(f_observed) * 1e-14)
            fpr_val = (np.count_nonzero(null_distribution >= (f_observed - tol)) + 1) / (n_permutations + 1)
            return fpr_val

        # Drop non-numerical columns
        panel = self.panel.copy().drop(columns=['ID', 'Level'], errors='ignore')

        # Create dictionary to store observed f values and fdr
        fdrs = {}
        eff_sizes = {}

        # Create a list with variable names
        features = panel.columns
        for feature in features:
            f_observed, null_distribution = process_feature(feature, n_permutations=n_permutations)
            fdr_value = fdr(null_distribution, f_observed, n_permutations)
            fdrs[feature] = fdr_value
            eff_sizes[feature] = f_observed

        # Convert dictionaries to pd.DataFrames
        fdrs = pd.DataFrame([fdrs])
        eff_sizes = pd.DataFrame([eff_sizes])
        return fdrs, eff_sizes

    def calculate_roc(self, feature):
        """
        df (dataframe with raw feature data) - pd.DataFrame
        feature (name of the feature for which the roc curve is being drawn) - string
        exclude (list containing the order in which a group within a factor is removed for comparisons two-by-two) - list
        """
        df = self.panel.copy()

        groups = df['Level'].unique()
        group_pairs = list(combinations(groups, 2))

        # Helper function to encode the positive class dynamically
        def encode_positive(sub_df, score_col=feature):
            # Compute mean score for each group
            means = sub_df.groupby('Level')[score_col].mean()

            # Identify the group with the highest mean score
            positive_group = means.idxmax()

            # Create a binary label where the positive group gets 1 and the other gets 0
            sub_df['binary_label'] = (sub_df['Level'] == positive_group).astype(int)
            return sub_df

        results = []

        for g1, g2 in group_pairs:
            sub_df = df[df['Level'].isin([g1, g2])].copy()
            sub_df = encode_positive(sub_df)

            y_true = sub_df['binary_label'].values
            y_scores = sub_df[feature].values

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_value = auc(fpr, tpr)

            results.append({
                'pair': f'{g1}_vs_{g2}',
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc_value
            })
        return pd.DataFrame(results)

    def draw_roc(self, feature):
        """
        Draw ROC curve for a single actigraphy-derived feature.
        """
        # Calculate parameters for the ROC curve
        df = self.calculate_roc(feature)

        for _,row in df.iterrows():
            # Plot ROC curve
            plt.plot(row['fpr'],
                     row['tpr'],
                     label=f"{row['pair']} (AUC = {row['auc']:.2f})",
                     linewidth=2,
                     alpha=0.8)

        # Plot random line
        plt.plot([0, 1], [0, 1], color='black', alpha=0.9, linestyle="--", label='chance', linewidth=2)

        # Label the plot
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(feature, fontsize=16)
        plt.legend(loc='lower right', frameon=True, fontsize=12)
        plt.show()