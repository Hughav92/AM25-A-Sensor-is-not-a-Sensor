"""
utils.py
--------
Utility functions for signal processing, data manipulation, and analysis used throughout the datalogger system.

Includes filtering, interpolation, envelope extraction, regression, quaternion math, and more.
"""

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter, savgol_filter, hilbert, correlate
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def vector_magnitude(x, y, z):

    return np.sqrt(x**2 + y **2 + z**2)

def crop_dataframes(dataframe, start_time, end_time):
    
    if 'receive_timestamp' not in dataframe.columns:
        raise ValueError("Each DataFrame must contain a 'receive_timestamp' column.")
    
    dataframe = dataframe.sort_values(by='receive_timestamp')
    
    nearest_start_idx = (dataframe['receive_timestamp'] - start_time).abs().idxmin()
    nearest_end_idx = (dataframe['receive_timestamp'] - end_time).abs().idxmin()
    
    start_idx, end_idx = sorted([nearest_start_idx, nearest_end_idx])
    
    cropped_df = dataframe.iloc[start_idx:end_idx + 1]
    
    return cropped_df

def acceleration_from_position(times, x, y, z):
    # Compute first derivatives (velocity)
    vx = np.gradient(x, times)
    vy = np.gradient(y, times)
    vz = np.gradient(z, times)
    
    # Compute second derivatives (acceleration)
    ax = np.gradient(vx, times)
    ay = np.gradient(vy, times)
    az = np.gradient(vz, times)
    
    # Compute acceleration magnitude
    acceleration_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    
    return acceleration_magnitude

def acceleration_magnitude_from_position_no_times(x, y, z):

    vx = np.diff(x)
    vy = np.diff(y)
    vz = np.diff(z)

    ax = np.diff(vx)
    ay = np.diff(vy)
    az = np.diff(vz)

    acceleration_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    
    return acceleration_magnitude

def g_to_mm_s2(g):
    conversion_factor = 9.81 * 1000
    return g * conversion_factor

def m_s2_to_g(m_s2):
    conversion_factor = 1 / 9.81
    return m_s2 * conversion_factor

def linear_interpolate_1d(arr, new_length):

    old_indices = np.linspace(0, len(arr) - 1, len(arr))
    
    new_indices = np.linspace(0, len(arr) - 1, new_length)
    
    interpolated_array = np.interp(new_indices, old_indices, arr)
    
    return interpolated_array

def sine_wave(t, A, f, phase, offset):
    return A * np.sin(2 * np.pi * f * t + phase) + offset

def find_zero_crossing(arr):
    for i in range(1, len(arr)):
        if (arr[i-1] > 0 and arr[i] < 0) or (arr[i-1] < 0 and arr[i] > 0):
            return i-1
    return None

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filter to pass frequencies between lowcut and highcut
    
    Parameters:
    data (array-like): The input signal to be filtered.
    lowcut (float): The lower cutoff frequency in Hz.
    highcut (float): The upper cutoff frequency in Hz.
    fs (float): The sampling rate of the signal in Hz.
    order (int): The order of the filter (default is 4).
    
    Returns:
    filtered_data (array-like): The filtered signal.
    """
    
    # Normalize the cutoff frequencies by the Nyquist frequency
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Get the filter coefficients
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def bandpass_savgol_filter(data, lowcut, highcut, fs, window_length=51, polyorder=3):
    """
    Apply a Savitzky-Golay filter to approximate bandpass filtering behavior.
    
    Parameters:
    data (array-like): The input signal to be filtered.
    lowcut (float): The lower cutoff frequency in Hz.
    highcut (float): The upper cutoff frequency in Hz.
    fs (float): The sampling rate of the signal in Hz.
    window_length (int): The length of the filter window (default is 51).
    polyorder (int): The order of the polynomial used (default is 3).
    
    Returns:
    filtered_data (array-like): The filtered signal.
    """
    
    # Create a time vector to simulate the bandpass effect in the frequency domain
    nyquist = 0.5 * fs
    freq_range = np.fft.fftfreq(len(data), d=1/fs)
    mask = (np.abs(freq_range) >= lowcut) & (np.abs(freq_range) <= highcut)
    
    # Apply FFT to the input data
    data_fft = np.fft.fft(data)
    
    # Zero out frequencies outside of the bandpass range
    data_fft[~mask] = 0
    
    # Perform inverse FFT to get the filtered signal
    filtered_data_freq_domain = np.fft.ifft(data_fft)
    
    # Apply Savitzky-Golay filter to smooth the data
    filtered_data = savgol_filter(np.real(filtered_data_freq_domain), window_length, polyorder)
    
    return filtered_data

def amplitude_envelope(signal):
    """
    Compute the amplitude envelope of a signal using the Hilbert transform.

    Parameters:
    signal (np.ndarray): The input signal.

    Returns:
    np.ndarray: The amplitude envelope of the input signal.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def rms_envelope(signal, window_size):
    """
    Compute the RMS-based amplitude envelope of a signal.

    Parameters:
    signal (np.ndarray): The input signal.
    window_size (int): The size of the moving window for RMS calculation.

    Returns:
    np.ndarray: The RMS amplitude envelope of the input signal.
    """
    squared_signal = signal ** 2
    window = np.ones(window_size) / window_size
    
    # Pad the signal to avoid edge effects
    pad_size = window_size // 2
    padded_signal = np.pad(squared_signal, (pad_size, pad_size), mode='edge')
    
    # Apply convolution and take the square root
    rms_signal = np.sqrt(np.convolve(padded_signal, window, mode='valid'))
    
    return rms_signal

def calculate_periodogram(signal, fs):
    """
    Compute the periodogram of a given signal.
    
    Parameters:
        signal (numpy array): The input signal.
        fs (float): Sampling frequency of the signal.
    
    Returns:
        freqs (numpy array): Frequencies corresponding to the periodogram.
        psd (numpy array): Power spectral density of the signal.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    psd = (np.abs(fft_result) ** 2) / n  # Power Spectral Density (PSD)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    return freqs[:n // 2], psd[:n // 2]  # Return positive frequencies only

def plot_periodogram(freqs, psd):
    """
    Plot the periodogram.
    
    Parameters:
        freqs (numpy array): Frequencies.
        psd (numpy array): Power spectral density.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, psd, color='b', linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Periodogram')
    plt.grid(True)
    plt.show()

def fit_linear_regression(time_series):
    """
    Fits a linear regression model to a time series.
    
    Parameters:
    time_series (np.ndarray): A numpy array of observations.
    
    Returns:
    dict: A dictionary containing the model, predictions, and coefficients.
    """
    if not isinstance(time_series, np.ndarray):
        raise ValueError("Input time_series must be a numpy array")
    
    # Convert index to numerical values
    X = np.arange(len(time_series)).reshape(-1, 1)  # Time steps as feature
    y = time_series.reshape(-1, 1)  # Observations
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, label="Actual Data")
    plt.plot(X, predictions, label="Linear Fit", linestyle='dashed')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Linear Regression Fit to Time Series")
    plt.show()
    
    return {
        "model": model,
        "predictions": predictions,
        "slope": model.coef_[0][0],
        "intercept": model.intercept_[0]
    }

def correlation(signal1, signal2):
    """
    Computes the cross-correlation between two signals using scipy.signal.correlate
    and returns the correlation values along with the corresponding lags.
    
    Parameters:
    signal1 (array-like): First input signal.
    signal2 (array-like): Second input signal.
    
    Returns:
    lags (numpy array): The lag values corresponding to the correlation values.
    correlation (numpy array): The cross-correlation values.
    """
    # Ensure signals are numpy arrays
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    # Compute cross-correlation using scipy
    correlation = correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')

    # Normalize the correlation values
    correlation /= (np.std(signal1) * np.std(signal2) * len(signal1))

    # Compute lag values
    lags = np.arange(-len(signal1) + 1, len(signal2))

    return lags, correlation

def first_maximum_lag(corr, lags, threshold_scale=0.5):

    corr_max = np.max(corr)
    threshold = corr_max*threshold_scale
    
    for i in range(1, len(corr) - 1):
        if corr[i] >= threshold:
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1]:
                return lags[i]
            
def quart_to_eul(w, x, y, z, angle="deg"):

    # norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    # w, x, y, z = w / norm, x / norm, y / norm, z / norm
    
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    if angle == "deg":
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return roll, pitch, yaw

def quart_to_ang_vel(q1, q2, sr):

    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)

    q_rel = q2*q1.inv()

    omega = q_rel.as_rotvec()/sr

    return omega


