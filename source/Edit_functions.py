import numpy as np


def calculate_doppler_shift_frequency(signal, sample_rate):
    # Вычисление автокорреляции сигнала
    autocorr = np.correlate(signal, signal, mode='full')

    # Нахождение положительного пика автокорреляции (больше нуля)
    positive_peaks = np.where(autocorr > 0)[0]

    # Нахождение первого положительного пика, отличного от нуля
    peak_index = positive_peaks[0] if positive_peaks.any() else None

    if peak_index is not None:
        # Вычисление частоты Доплеровского сдвига
        frequency_bins = len(signal) * 2  # Количество бинов в автокорреляции
        doppler_shift_frequency = (peak_index - len(signal)) / (2.0 * len(signal)) * sample_rate / frequency_bins
    else:
        doppler_shift_frequency = None

    return doppler_shift_frequency


def subtract_last_detection_time(df, detection_time):
    if len(df) > 0:
        last_detection_time = df.iloc[-1]["Время обнаружения филамента, мс"]
        result = round((detection_time - last_detection_time) * 1000)
    else:
        # Handle the case when the DataFrame is empty
        result = None
    return result
