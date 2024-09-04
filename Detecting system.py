import gc
import os
import time

import pandas as pd
from keras.models import load_model

from Files_operating import read_dataFile, save_results_toFiles
from source.NN_enviroments import *
from source.Signal_processing import data_converting_CNN, \
    fft_butter_skewness_filtering


def get_channel(data):
    # Предложение пользователю выбрать канал
    available_channels = [col for col in data.columns if col != "t" and str(data[col][0]) != 'nan']
    selected_channels = input("\nДоступные каналы: " + ' '.join(
        available_channels) + "\n----------\nВведите каналы для анализа: ").strip().split()
    # Проверка наличия выбранного канала в данных
    if all([selected_channel in available_channels for selected_channel in selected_channels]) and len(selected_channels) > 0:
        # log
        print(f"\n#log: Каналы {', '.join(selected_channels)} считаны успешно")

        return selected_channels
    else:
        print("Некоторые каналы не найдены в данных.")
        selected_channels = input("Доступные каналы: " + ' '.join(
            available_channels) + "\n----------\nВведите каналы для анализа: ").strip().split()

    if all([selected_channel in available_channels for selected_channel in selected_channels]) and len(selected_channels) > 0:
        # log
        print(f"\n#log: Каналы {', '.join(selected_channels)} считаны успешно")

        return selected_channels
    return None


def filtering_function(fragments, name_filter, path_to_proj):
    fragments_smooth = data_converting_CNN(fragments)
    # log
    print("#log: Данные фрагментов нормализованы и подготовлены для нейро-фильтра.")

    gc.collect()

    # neuro-filter
    neuro_filter = load_model(path_to_proj + f"models/{name_filter}.keras", safe_mode=False,
                              custom_objects={"focal_loss": focal_loss_01,
                                              "focal_loss_01": focal_loss_01,
                                              "focal_crossentropy": focal_crossentropy,
                                              "f_m": f_m,
                                              "f1_m": f1_m,
                                              "precision_m": precision_m,
                                              "recall_m": recall_m})

    # log
    print("#log: Версия фильтра:", name_filter)
    print("#log: Запуск фильтра. Прогнозирование:")
    start = time.time()
    predictions = np.apply_along_axis(class_scores_processing, 1, neuro_filter.predict(fragments_smooth, verbose=1))
    # log
    print(f"#log: Обработка завершена. Tooks - {round(time.time() - start, 5) * 1000} ms.\n")

    gc.collect()
    return predictions


def detect_function(data_t, data_ch, file_name, signal_meta, signal_channels, path_to_proj, path_to_csv):
    # log
    print("\n#log: Начата предварительная обработка данных.")
    start = time.time()
    fragments = fft_butter_skewness_filtering(data_t, data_ch, SIGNAL_RATE, f_disp=True)
    # log
    print(
        f"#log: Предварительная обработка и фильтрация выполнена успешно. Tooks - {round(time.time() - start, 2) * 1} s.")
    print("==========================================")
    print(f"#log: Количество найденных фрагментов: {len(fragments[0])}")
    print("==========================================")

    # function for work with NN
    for ch_i in range(len(signal_channels)):
        name_filters = ["cnn_bin_class_13"]  # , "auto_bin_class_12"
        # "auto_bin_class_8", "auto_bin_class_11", "cnn_bin_class_4", "cnn_bin_class_10",

        for name_filter in name_filters:
            predictions = filtering_function([fragments[0], fragments[ch_i+1]], name_filter, path_to_proj)

            edge = 0.75
            f_saving = False

            f_plot = input("\nВедите у, чтобы отобразить кривую распределения результатов: ")
            if f_plot.lower() in ["y", "у", "e", "н"]:
                plot_predictionCurve(predictions)

            # try:
            #     edge = float(input("\nВедите граничное значение для оценки филаментов (разделитель - '.'): "))
            # except Exception as e:
            #     pass

            filtered = predictions >= edge
            # log
            print(f"\n#log: Обработка завершена. Проведена оценка с границей: {edge}")

            # log
            print("==========================================")
            print(f"#log: Количество спрогнозированных филаментов: {len(list(filter(lambda x: x, filtered)))}")
            print("==========================================")

            if f_saving or input("\nВедите у, чтобы запустить процесс сохранения: ").lower() in ["y", "у", "e", "н"]:
                # f_save = input("\nВедите у, чтобы сохранить все фрагменты (без фильтрации по оценке): ")
                f_save_all = False
                add_name_str = "fil_"
                # if f_save.lower() in ["y", "у", "e", "н"]:
                #     f_save_all = True
                #     add_name_str = "all_"

                if not os.path.exists(path_to_csv + "result_data/"):
                    os.mkdir(path_to_csv + "result_data/")

                data_csv_name = f"result_data/{file_name[:-4]}_{name_filter}_result_{add_name_str}all_ch_data.csv"

                # log
                print("\n#log: Сохранение результатов.")
                start = time.time()
                signal_meta["ch"] = signal_channels[ch_i]
                save_results_toFiles(predictions, [fragments[0], fragments[ch_i + 1]], data_csv_name, signal_meta,
                                     path_to_csv=path_to_proj + path_to_csv, edge=edge,
                                     f_save_all=f_save_all, f_disp=True)

                # remove
                ind_sec_ch = abs(ch_i - 1)  # remove
                signal_meta["ch"] = signal_channels[ind_sec_ch]  # remove + "_0"
                save_results_toFiles(predictions, [fragments[0], fragments[ind_sec_ch + 1]], data_csv_name, signal_meta,  # remove
                                     path_to_csv=path_to_proj + path_to_csv, edge=edge,   # remove
                                     f_save_all=f_save_all, f_disp=True)  # remove

                # log
                print(f"#log: Результаты сохранены. Tooks - {round(time.time() - start, 2) * 1} s. Файлы:\n" +
                      f"{path_to_proj + path_to_csv + data_csv_name}\n")
                gc.collect()


proj_path = input("Введите путь к запускаемому файлу (DBS-processing/): ")
data_path = input("Введите путь к файлам с данными относительно запускаемого файла (data_csv/): ")

if not os.path.exists(data_path):
    os.mkdir(data_path)

filename = input("Введите имя файла. Доступные файлы:\n" + "\n".join(
    list(filter(lambda x: '.dat' in x or '.txt' in x, os.listdir(data_path)))) + "\n----------\n")

file_path = data_path + filename
FILE_D_ID = filename[:5]  # "00000"
# log
print(f"\n#log: Выбран файл {filename} (FILE_ID: {FILE_D_ID})")

# filename = "41226 DBS_fragment_for_DA_analytic.csv"  # remove
# FILE_D_ID = filename[:5] + "_fr"  # remove
# file_path = data_path + filename  # remove

start = time.time()
# df = pd.read_csv(file_path)  # remove
df = read_dataFile(file_path, proj_path)

# log
print(f"#log: Файл {filename} считан успешно. Tooks - {round(time.time() - start, 2) * 1} s.")
gc.collect()

SIGNAL_RATE = float(input("\nВведите частоту дискретизации для данного сигнала (4 / 10): "))  # 4
signal_maxLength = 512

channels = get_channel(df)
while channels is not None:
    meta = {"id": FILE_D_ID, "rate": SIGNAL_RATE}
    selected_data = df[["t"] + channels].astype({"t": "float64"})

    # Выбираем область сигнала
    x = np.array(selected_data.t[(selected_data.t > -np.inf) * (selected_data.t < np.inf)])
    y = []
    for ch in channels:
        y.append(np.array(selected_data[ch][(selected_data.t > -np.inf) * (selected_data.t < np.inf)]))

    detect_function(x, y, filename, meta, channels, proj_path, data_path)
    gc.collect()

    channels = get_channel(df)
