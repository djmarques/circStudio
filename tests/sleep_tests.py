import pandas as pd
import numpy as np

def main():
    sleep_onset_latency_test()
    pd.Timedelta(hours=6) / pd.Timedelta(hours=8)


def sleep_onset_latency_test():
    """
    Computes sleep onset latency using the Roenneberg algorithm to predict sleep onset and
    the sleep diary to determine total bedtime.

    Parameters
    ----------

    Returns
    -------
    pd.Series
        Sleep onset latency.

    """
    main_sleep_df = pd.DataFrame({
        'date': ['1918-01-24', '1918-01-25', '1918-01-27'],
        'start_time': ['1918-01-24 23:43:00', '1918-01-25 22:17:00', '1918-01-27 01:43:00'],
        'stop_time': ['1918-01-25 06:25:00', '1918-01-26 07:20:00', '1918-01-27 07:11:00']
    })
    for col in main_sleep_df.columns:
        main_sleep_df[col] = pd.to_datetime(main_sleep_df[col])

    diary_nights_df = pd.DataFrame({
        'START': ['1918-01-24 23:00:00', '1918-01-25 22:00:00', '1918-01-27 00:00:00'],
        'END': ['1918-01-25 07:00:00', '1918-01-26 07:30:00', '1918-01-27 07:30:00']
    })
    for col in diary_nights_df.columns:
        diary_nights_df[col] = pd.to_datetime(diary_nights_df[col])

    sol = {}

    for idx, row in diary_nights_df.iterrows():
        date = row['START']
        matches = main_sleep_df[main_sleep_df['start_time'].dt.date == date.date()]
        if not matches.empty:
            sleep_onset = matches.iloc[0]['start_time']
            latency = sleep_onset - row['START']
            sol[date.date()] = latency
    sol = pd.Series(sol)
    return pd.Series(sol), np.mean(sol)


if __name__ == '__main__':
    main()