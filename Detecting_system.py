import gc
import os
import time
import sys

import pandas as pd
from tensorflow.keras.models import load_model

from Files_operating import read_dataFile, save_results_toFiles
from source.NN_enviroments import *
from source.Signal_processing import data_converting_CNN, \
    fft_butter_skewness_filtering


def filtering_function(fragments, name_filter, path_to_proj):
    if len(fragments[0]) == 0:
        return None
    
    fragments_smooth = data_converting_CNN(fragments)
    # log
    print("#log: Данные фрагментов нормализованы и подготовлены для нейро-фильтра.")

    gc.collect()

    # neuro-filter
    neuro_filter = load_model(path_to_proj + f"models/{name_filter}.keras",
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


def detect_function(data_t, data_ch, file_name, signal_meta, signal_channels, path_to_proj, path_to_csv, nn_filename, SIGNAL_RATE):
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

    if len(fragments[0]) == 0:
        print("\n")
        return 
    
    # single fragment analysis with NN
    for ch_i in range(len(signal_channels)):
        name_filter = nn_filename

        predictions = filtering_function([fragments[0], fragments[ch_i+1]], name_filter, path_to_proj)
        
        if predictions is None:
            print("\n")
            continue

        edge = 0.75
    
        filtered = predictions >= edge
        # log
        print(f"\n#log: Обработка завершена. Проведена оценка с границей: {edge}")

        # log
        print("==========================================")
        print(f"#log: Количество спрогнозированных филаментов: {len(filtered)}")
        print("==========================================")

        if len(filtered) == 0:
            print("\n")
            continue

        f_save_all = False

        if not os.path.exists(path_to_csv + "result_data/"):
            os.mkdir(path_to_csv + "result_data/")
        if not os.path.exists(path_to_csv + "result_fragments/"):
            os.mkdir(path_to_csv + "result_fragments/")

        data_csv_name = f"result_data/{file_name[:-4]}_{name_filter}_data.csv"

        # log
        print("\n#log: Сохранение результатов.")
        start = time.time()
        signal_meta["ch"] = signal_channels[ch_i]
        save_results_toFiles(predictions, [fragments[0], fragments[ch_i + 1]], data_csv_name, signal_meta,
                             path_to_csv=path_to_proj + path_to_csv, edge=edge,
                             f_save_all=f_save_all, f_disp=True)

        # remove
        ind_sec_ch = abs(ch_i - 1)  # remove
        signal_meta["ch"] = signal_channels[ind_sec_ch] + "_0"  # remove
        save_results_toFiles(predictions, [fragments[0], fragments[ind_sec_ch + 1]], data_csv_name, signal_meta,  # remove
                             path_to_csv=path_to_proj + path_to_csv, edge=edge,   # remove
                             f_save_all=f_save_all, f_disp=True)  # remove

        # log
        print(f"#log: Результаты сохранены. Tooks - {round(time.time() - start, 2) * 1} s. Файлы:\n" +
              f"{path_to_proj + path_to_csv + data_csv_name}\n")
        gc.collect()


def init_proc(filename, nn_name, SIGNAL_RATE, input_channels):
    proj_path = ""
    data_path = "data_csv/"

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    file_path = data_path + filename
    FILE_D_ID = filename[:5]  # "00000"
    # log
    print(f"\n#log: Выбран файл {filename} (FILE_ID: {FILE_D_ID})")

    start = time.time()
    df = read_dataFile(file_path, proj_path, id_file=FILE_D_ID)

    # log
    print(f"#log: Файл {filename} считан успешно. Tooks - {round(time.time() - start, 2) * 1} s.")
    gc.collect()
    
    signal_maxLength = 512

    for i in range(0, len(input_channels) - len(input_channels) % 2, 2):
        channels = [input_channels[i], input_channels[i + 1]]  # get_channel(df, input_channels)
        available_channels = [col for col in df.columns if col != "t" and str(df[col][0]) != 'nan']
        if not all([selected_channel in available_channels for selected_channel in channels]):
            print(f"Некоторые каналы не найдены в данных ({channels}).")
            return
        else:
            print(f"\n#log: Для файла FILE_ID: {FILE_D_ID} выбраны каналы: {channels}")
        
        meta = {"id": FILE_D_ID, "rate": SIGNAL_RATE}
        selected_data = df[["t"] + channels].astype({"t": "float64"})
    
        # Выбираем область сигнала
        x = np.array(selected_data.t[(selected_data.t > -np.inf) & (selected_data.t < np.inf)])
        y = []
        for ch in channels:
            y.append(np.array(selected_data[ch][(selected_data.t > -np.inf) & (selected_data.t < np.inf)]))
    
        detect_function(x, y, filename, meta, channels, proj_path, data_path, nn_name, SIGNAL_RATE)
        gc.collect()


if __name__ == "__main__" and not (sys.stdin and sys.stdin.isatty()):
    # get args from CL
    print("Sys args (filename | NN filename | SIGNAL_RATE | ch1 ch2 ...):\n", sys.argv)
    init_proc(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:])

else:
    print("Program is supposed to run out from command line.")

