import warnings

import numpy as np

warnings.filterwarnings("ignore")

import gc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_SETTINGS"] = "false"

import time
import datetime
from multiprocessing import Pool

import pandas as pd
import scipy.interpolate as sc_i
from keras.models import load_model

from Files_operating import read_dataFile, save_df_toFile
from source.NN_enviroments import *
from source.Signal_processing import data_converting_CNN, \
    fft_butter_skewness_filtering


def detect_function(data_t, data_ch, signal_meta, signal_channels, path_to_proj, path_to_csv):
    edge = 0.75

    start_time = time.time()
    fragments = fft_butter_skewness_filtering(data_t, data_ch, signal_meta["rate"], f_disp=True)

    # log
    print(
        f"#log: Preproc {', '.join(signal_channels)} is done. Took - {datetime.timedelta(seconds=int(time.time() - start_time))}")

    fragments_df = pd.DataFrame(columns=(['D_ID', 'Ch', 'Left', 'Right', 'Y', 'Length', 'Rate'] + [str(i) for i in
                                                                                                   range(signal_meta[
                                                                                                             "max_len"])]))

    filter_path = signal_meta["filter_path"]
    name_filter = filter_path.split("/")[-1].split(".")[0]
    file_data_csv_name = f"result_data/{signal_meta['id']}_{name_filter}_result_{' '.join(signal_channels)}.csv"
    file_ch_data_name = f"result_data/{signal_meta['id']}_{name_filter}_stats_{' '.join(signal_channels)}.txt"

    if not os.path.exists(path_to_csv + file_ch_data_name):
        with open(path_to_csv + file_ch_data_name, "a") as file:
            file.write(f"Stats of file {signal_meta['id']}.\nAnomaly fragments: {len(fragments[0])}\n")
            file.close()
    else:
        with open(path_to_csv + file_ch_data_name, "a") as file:
            file.write(f"\n{', '.join(signal_channels)}:\nAnomaly fragments: {len(fragments[0])}\n")
            file.close()

    fragments_num = len(fragments[0])
    start_time = time.time()

    fragments_smooth_1 = data_converting_CNN([fragments[0], fragments[1]])
    fragments_smooth_2 = data_converting_CNN([fragments[0], fragments[2]])

    neuro_filter = load_model(path_to_proj + filter_path, safe_mode=False,
                              custom_objects={"focal_loss": focal_loss_01,
                                              "focal_loss_01": focal_loss_01,
                                              "focal_crossentropy": focal_crossentropy,
                                              "f_m": f_m,
                                              "f1_m": f1_m,
                                              "precision_m": precision_m,
                                              "recall_m": recall_m})

    predict_classes_1 = neuro_filter.predict(fragments_smooth_1, verbose=1)
    scores_1 = np.round(np.apply_along_axis(lambda x: class_scores_processing(x), 1, predict_classes_1), 3)

    predict_classes_2 = neuro_filter.predict(fragments_smooth_2, verbose=1)
    scores_2 = np.round(np.apply_along_axis(lambda x: class_scores_processing(x), 1, predict_classes_2), 3)

    with open(path_to_csv + file_ch_data_name, "a") as file:
        file.write(f"\t{signal_channels[0]}:\n" +
                   f"\t\tCount classes: 0 - {scores_1[scores_1 < edge].shape[0]}, 1 - {scores_1[scores_1 >= edge].shape[0]};\n" +
                   f"\t\tMean scores: 0 - {predict_classes_1[scores_1 < edge][:, 0].mean()}, 1 - {scores_1[scores_1 >= edge].mean()}\n")
        file.write(f"\t{signal_channels[1]}:\n" +
                   f"\t\tCount classes: 0 - {scores_2[scores_2 < edge].shape[0]}, 1 - {scores_2[scores_2 >= edge].shape[0]};\n" +
                   f"\t\tMean scores: 0 - {predict_classes_2[scores_2 < edge][:, 0].mean()}, 1 - {scores_2[scores_2 >= edge].mean()}\n")
        file.close()

    if not os.path.exists("result_data/"):
        os.mkdir("result_data/")

    pred_index = 0
    TT_count = 0
    for i in range(fragments_num):
        fragment_x = fragments[0][i].tolist()
        fragment_values_1 = fragments[1][i].tolist()
        fragment_values_2 = fragments[2][i].tolist()

        # получение границ фрагмента из введённой строки
        fragment_range = [min(fragment_x), max(fragment_x)]

        # нормировка границ (если выходит за рамки сигнала) и получение его длительности
        fragment_range = [fragment_range[0], fragment_range[1]]
        fragment_length = fragment_range[1] - fragment_range[0]

        if fragment_length <= 0 or len(fragment_x) <= 5 or len(fragment_x) >= 500:
            continue

        # получение метки фрагмента из прогнозированных данных
        fragment_mark_1 = scores_1[pred_index]
        fragment_mark_2 = scores_2[pred_index]

        pred_index += 1

        if fragment_mark_1 >= edge or fragment_mark_2 >= edge:
            # получение с помощью интерполяции квадратичным сплайном нужного количества точек фрагмента (510 точек)
            fragment_interpolate_values_1 = sc_i.interp1d(fragment_x, fragment_values_1, kind="quadratic")(
                np.linspace(fragment_x[0], fragment_x[-1], signal_meta["max_len"]))
            fragment_interpolate_values_2 = sc_i.interp1d(fragment_x, fragment_values_2, kind="quadratic")(
                np.linspace(fragment_x[0], fragment_x[-1], signal_meta["max_len"]))

            # добавление данных в Data Frame
            fragments_df.loc[-1] = [signal_meta["id"], signal_meta["ch"][0], min(fragment_range),
                                    max(fragment_range),
                                    fragment_mark_1,
                                    fragment_length, signal_meta["rate"]] + list(
                fragment_interpolate_values_1)  # adding a row
            fragments_df.index = fragments_df.index + 1  # shifting index
            fragments_df = fragments_df.sort_index()  # sorting by index

            # добавление данных в Data Frame
            fragments_df.loc[-1] = [signal_meta["id"], signal_meta["ch"][1], min(fragment_range),
                                    max(fragment_range),
                                    fragment_mark_2,
                                    fragment_length, signal_meta["rate"]] + list(
                fragment_interpolate_values_2)  # adding a row
            fragments_df.index = fragments_df.index + 1  # shifting index
            fragments_df = fragments_df.sort_index()  # sorting by index

        if fragment_mark_1 >= edge and fragment_mark_2 >= edge:
            TT_count += 1

    fragments_df = fragments_df.drop_duplicates()

    # сохранение Data Frame
    if len(fragments_df.count(axis="rows")) > 0:
        fragments_df = fragments_df.drop_duplicates()
        with open(path_to_csv + file_ch_data_name, "a") as file:
            file.write(f"Total saved fragments: {fragments_df.shape[0] // 2} (x2 channels)\n" +
                       f"Two-True fragments: {TT_count} (x2 channels)\n")
            file.close()
        save_df_toFile(fragments_df, file_data_csv_name, path_to_csv)

    # log
    print(f"#log: Analysis & saving {', '.join(signal_channels)} is done. " +
          f"Took - {datetime.timedelta(seconds=int(time.time() - start_time))}")


def proceed_chs(data, s_meta, channels):
    channels = channels.split()
    s_meta["ch"] = channels
    selected_data = data[["t"] + channels].astype({"t": "float64"})

    # Выбираем область сигнала
    x = np.array(selected_data.t[(selected_data.t > -np.inf) * (selected_data.t < np.inf)])
    y = []
    for ch in channels:
        y.append(np.array(selected_data[ch][(selected_data.t > -np.inf) * (selected_data.t < np.inf)]))

    detect_function(x, y, s_meta, channels, s_meta["p_path"], s_meta["d_path"])


if __name__ == '__main__':
    # log
    print(f"Current time - {str(datetime.datetime.now()).split('.')[0]}\n")

    start_tot = time.time()

    proj_path = input("Введите путь к запускаемому файлу (Plasma_processing/): ")
    data_path = input("Введите путь к файлам с данными относительно запускаемого файла (data_csv/): ")
    neuro_path = "models/cnn_bin_class_13.keras"  # input("Введите путь к модели нейрофильтра относительно запускаемого файла (models/cnn_bin_class_13.keras): ")

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    filename = input("Введите имя файла. Доступные файлы:\n" + "\n".join(
        list(filter(lambda x: '.dat' in x or '.txt' in x, os.listdir(data_path)))) + "\n----------\n")

    file_path = data_path + filename
    FILE_D_ID = filename[:5]  # "00000"

    # log
    print(f"\n#log: Выбран файл {filename} (FILE_ID: {FILE_D_ID})")

    start = time.time()
    df = read_dataFile(file_path, proj_path)
    df.t = np.linspace(df.t.min(), df.t.max(), df.t.shape[0])

    # log
    print(f"#log: Файл {filename} считан успешно. Tooks - {datetime.timedelta(seconds=int(time.time() - start))}")
    gc.collect()

    SIGNAL_RATE = float(input("\nВведите частоту дискретизации для данного сигнала (4 / 10): "))  # 4
    signal_maxLength = 512

    available_channels = [col for col in df.columns if col != "t" and str(df[col][0]) != 'nan']
    n_pairs = len(available_channels) // 2

    for step in range(n_pairs // 10 + 1):
        n_pools = 10
        d_step = step * 10

        if n_pairs < step * 10 + n_pools:
            n_pools = n_pairs % 10
            d_step = (step - 1) * 10 if n_pairs < step * 10 else d_step

        pair_channels = [f"{available_channels[i + d_step]} {available_channels[i + d_step + 1]}" for i in
                         range(0, n_pools * 2, 2)]

        meta = {"id": FILE_D_ID, "rate": SIGNAL_RATE, "p_path": proj_path, "d_path": data_path, "filename": filename,
                "filter_path": neuro_path, "max_len": signal_maxLength}
        with Pool(n_pools) as p:
            input_args = [(df, meta, ch) for ch in pair_channels]
            p.starmap(proceed_chs, input_args)
        gc.collect()

    pair_channels = [f"{available_channels[i]} {available_channels[i + 1]}" for i in
                     range(0, len(available_channels), 2)]

    file_ch = f"result_data/{FILE_D_ID}_{neuro_path.split('/')[-1].split('.')[0]}_stats.txt"

    with open(proj_path + data_path + file_ch, "a") as file:
        text = f"Stats of file {FILE_D_ID}.\n"

        f_translate_ch = False
        translate_ch_str = input("\nВедите у, чтобы добавить каналам частоты зондирования: ")

        if translate_ch_str.lower() in ["y", "у", "e", "н"]:
            f_translate_ch = True
            db_tot = pd.DataFrame(columns=(['D_ID', 'Ch', 'Ch_Freq', 'Left', 'Right', 'Y', 'Length', 'Rate'] +
                                           [str(i) for i in range(signal_maxLength)]))
        else:
            db_tot = pd.DataFrame(columns=(['D_ID', 'Ch', 'Left', 'Right', 'Y', 'Length', 'Rate'] +
                                           [str(i) for i in range(signal_maxLength)]))

        for ch in pair_channels:
            if f_translate_ch:
                print("Введите частоту зондирования (в ГГц), соответствующую каналам")
                channels_freq = float(input(f"{ch}: ").strip())

            name_stat = f"result_data/{FILE_D_ID}_{neuro_path.split('/')[-1].split('.')[0]}_stats_{ch}.txt"
            name_pd = f"result_data/{FILE_D_ID}_{neuro_path.split('/')[-1].split('.')[0]}_result_{ch}.csv"

            db_1 = pd.read_csv(proj_path + data_path + name_pd)
            if f_translate_ch:
                db_1['Ch_Freq'] = channels_freq
            db_tot = pd.concat([db_tot, db_1], axis=0)

            text += f"\n{ch}:\n"
            text += "\n".join(open(proj_path + data_path + name_stat, "r").read().split("\n")[1:])

            os.remove(proj_path + data_path + name_stat)
            os.remove(proj_path + data_path + name_pd)
        file.write(text)
        file.close()

        db_tot.to_csv(proj_path + data_path +
                      f"result_data/{FILE_D_ID}_{neuro_path.split('/')[-1].split('.')[0]}_result_all_ch.csv",
                      index=False)

    # log
    print(f"#log: Файл {filename} обработан успешно. " +
          f"Total time - {datetime.timedelta(seconds=int(time.time() - start_tot))}. {datetime.datetime.now()}")
