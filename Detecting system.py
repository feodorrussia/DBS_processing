import gc
import os

import keras
import numpy as np
import pandas as pd
import scipy.interpolate as sc_i

from Signal_processing import *
from Files_operating import read_dataFile, save_results_toFiles


path_to_proj = ""  # Plasma_processing/
path_to_csv = "data_csv/"  #

if not os.path.exists(path_to_csv):
    os.mkdir(path_to_csv)

file_path = input("Введите имя файла. Доступные файлы:\n" + "\n".join(
    list(filter(lambda x: '.dat' in x or '.txt' in x, os.listdir(path_to_csv)))) + "\n----------\n")

file_name = file_path.split('/')[-1]
FILE_D_ID = file_name[:5]  # "00000"
print(f"Выбран файл {file_name} (FILE_ID: {FILE_D_ID})")

fragments_csv_name = file_path[:-4] + "_fragments.csv"

data = read_dataFile(file_path, path_to_proj)
gc.collect()

SIGNAL_RATE = float(input("Введите частоту дискретизации для данного сигнала: "))
signal_maxLength = 512

# Предложение пользователю выбрать канал
available_channels = [col for col in data.columns if col != "t" and str(data[col][0]) != 'nan']
selected_channel = input("Доступные каналы: " + ', '.join(available_channels) + "\nВыберите канал: ")

# Проверка наличия выбранного канала в данных
if selected_channel in available_channels:
    selected_data = data[["t", selected_channel]].astype({"t": "float64"})
else:
    print("Выбранный канал не найден в данных.")

# Выбираем область графика
start = -np.inf
end = np.inf

x = np.array(data.t[(data.t > start) * (data.t < end)])
y = np.array(data[selected_channel][(data.t > start) * (data.t < end)])

fragments = fft_butter_skewness_filtering(x, y)
fragments_smooth = data_converting_CNN(fragments)
gc.collect()

# neuro-filter
name_filter = "cnn_bin_class_4"
neuro_filter = keras.models.load_model(path_to_proj + f"models/{name_filter}.keras", safe_mode=False)

# custom_objects={"focal_crossentropy": focal_crossentropy,
#                                                        "f1_m": f1_m,
#                                                        "precision_m": precision_m,
#                                                        "recall_m": recall_m}
predictions = neuro_filter.predict(fragments_smooth)

gc.collect()

edge = 0.75
filtered = predictions >= edge

print("==========================================")
print(f"Количество найденных филаментов: {len(fragments[0])}")
print("==========================================")
print(f"Количество отобранных филаментов: {len(list(filter(lambda x: x, filtered)))}")
print("==========================================")

if not os.path.exists(path_to_csv + "result_data/"):
    os.mkdir(path_to_csv + "result_data/")

data_csv_name = f"new_{file_name[:-4]}_{name_filter}_result_data.csv"
fragments_csv_name = f"new_{file_name[:-4]}_{name_filter}_result_fragments.csv"

save_results_toFiles(predictions, fragments, data_csv_name, fragments_csv_name, {"id": FILE_D_ID, "ch": selected_channel, "rate": SIGNAL_RATE}, path_to_csv=path_to_csv)

gc.collect()
