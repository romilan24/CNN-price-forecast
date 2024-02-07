import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar

def date_and_hour(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['he'] = (df['datetime'].dt.hour + 1) % 25
    return df

def add_holiday_variable(data, date_column, start_date, end_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    data['is_holiday'] = (data[date_column].dt.normalize().isin(holidays)).astype(int)
    return data

def create_lagged_variables(data, lagged_column, lag_range):
    for lag in range(1, lag_range + 1):
        data[f'{lagged_column}_lag_{lag}'] = data[lagged_column].shift(lag)
    return data

def swap_missing_data(merged_df, sf_columns, sj_columns):
    for col_sf, col_sj in zip(sf_columns, sj_columns):
        merged_df[col_sf].fillna(merged_df[col_sj], inplace=True)
        merged_df[col_sj].fillna(merged_df[col_sf], inplace=True)
    return merged_df

def windowing(X_input, y_input, history_size):
    data = []
    labels = []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size: i, :])
        labels.append(y_input[i])
    return np.array(data), np.array(labels).reshape(-1, 1)

def rename_columns(df, column_mapping):
    df = df.rename(columns=column_mapping)
    return df

def interpolate_missing(df):
    result_df = df.copy()
    result_df['datetime'] = pd.to_datetime(result_df['datetime'])
    for column in df.columns:
        if column not in ['datetime'] and pd.api.types.is_numeric_dtype(result_df[column]):
            try:
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
                result_df[column] = result_df.groupby(result_df['datetime'].dt.hour)[column].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both'))
            except ValueError:
                print(f"Skipping interpolation for non-numeric column: {column}")
    return result_df

def split_data(df, test_size=0.2, random_state=35):
    X = df.drop(['caiso_load_actuals', 'datetime'], axis=1)
    y = df['caiso_load_actuals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100