import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from preprocess import rename_columns, swap_missing_data, interpolate_missing, add_holiday_variable, windowing

# Load and preprocess data
path = 'C:/Users/groutgauss/Machine_Learning_Projects/CAISO Price Forecast/Deep Learning/'
merged_df = pd.read_csv(path + 'data.csv')

# Rename column headers
column_mapping = {
    'Datetime': 'datetime',
    'Current demand': 'caiso_load_actuals',
    'KCASANFR698_Temperature': 'SF_temp',
    'KCASANFR698_Dew_Point': 'SF_dew',
    'KCASANFR698_Humidity': 'SF_humidity',
    'KCASANFR698_Speed': 'SF_windspeed',
    'KCASANFR698_Gust': 'SF_windgust',
    'KCASANFR698_Pressure': 'SF_pressure',
    'KCASANJO17_Temperature': 'SJ_temp',
    'KCASANJO17_Dew_Point': 'SJ_dew',
    'KCASANJO17_Humidity': 'SJ_humidity',
    'KCASANJO17_Speed': 'SJ_windspeed',
    'KCASANJO17_Gust': 'SJ_windgust',
    'KCASANJO17_Pressure': 'SJ_pressure',
    'KCABAKER271_Temperature': 'BAKE_temp',
    'KCABAKER271_Humidity': 'BAKE_humidity',
    'KCABAKER271_Speed': 'BAKE_windspeed',
    'KCABAKER271_Pressure': 'BAKE_pressure',
    'KCAELSEG23_Temperature': 'EL_temp',
    'KCAELSEG23_Dew_Point': 'EL_dew',
    'KCAELSEG23_Humidity': 'EL_humidity',
    'KCAELSEG23_Speed': 'EL_windspeed',
    'KCAELSEG23_Gust': 'EL_windgust',
    'KCAELSEG23_Pressure': 'EL_pressure',
    'KCARIVER117_Temperature': 'RIV_temp',
    'KCARIVER117_Dew_Point': 'RIV_dew',
    'KCARIVER117_Humidity': 'RIV_humidity',
    'KCARIVER117_Speed': 'RIV_windspeed',
    'KCARIVER117_Gust': 'RIV_windgust',
    'KCARIVER117_Pressure': 'RIV_pressure'
}

# Columns for swapping missing NaN data between SF and SJ
sf_columns = ['SF_temp', 'SF_dew', 'SF_humidity', 'SF_windspeed', 'SF_windgust', 'SF_pressure']
sj_columns = ['SJ_temp', 'SJ_dew', 'SJ_humidity', 'SJ_windspeed', 'SJ_windgust', 'SJ_pressure']

# Data preprocessing steps
data = rename_columns(merged_df, column_mapping)
data = swap_missing_data(data, sf_columns, sj_columns)
data = interpolate_missing(data)
data = add_holiday_variable(data, 'datetime', '2021-01-02', '2023-10-03')

# Split data into features and target
X = data.drop(['datetime', 'TH_SP15_GEN-APND'], axis=1).values
y = data['TH_SP15_GEN-APND'].values

# Apply PCA
pca = PCA(n_components=0.8)
scaler_pca = StandardScaler()
X_pca = pca.fit_transform(scaler_pca.fit_transform(X))

train_cutoff = int(0.8*X_pca.shape[0])
val_cutoff   = int(0.9*X_pca.shape[0])

scaler_y = MinMaxScaler()
scaler_y.fit(y[:train_cutoff].reshape(-1,1))
y_norm = scaler_y.transform(y.reshape(-1,1))

# Windowing for sequence data
hist_size = 24
X_windowed, y_windowed = windowing(X_pca, y, hist_size)

# Train-validation-test split
X_train, X_val_test, y_train, y_val_test = train_test_split(X_windowed, y_windowed)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Define the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X_train.shape[-2:]),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=125, batch_size=64, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1,1))
y_test_inv = scaler_y.inverse_transform(y_test)

def plot_results(y_pred_actual, y_test_inv, history, model_name):
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))

    # Prediction vs actual price chart
    ax[0].plot(y_pred_actual[:1000])
    ax[0].plot(y_test_inv[:1000])
    ax[0].legend(['prediction', 'actual'], loc='upper left')
    ax[0].set_title(f'Prediction vs actual price for 1000 observation in test set ({model_name})')
    ax[0].set_xlabel('Observation')
    ax[0].set_ylabel('Price')

    # MAE chart
    ax[1].plot(history.history['loss'], label='Training Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].legend()
    ax[1].set_title(f'Training and validation MAE ({model_name})')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAE')

    fig.tight_layout()
    plt.show()

print('')
print('')
print('---------------------------------------------------')
print(f'LSTM MAE for test set : {round(mean_absolute_error(y_pred,y_test),3)}')
print('---------------------------------------------------')
y_pred_actual = scaler_y.inverse_transform(y_pred)
print('')
plot_results(y_pred_actual, y_test_inv, history,'CNN')                                                   
                                                            