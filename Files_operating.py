import os

import numpy as np
import pandas as pd
import scipy.interpolate as sc_i


def clear_space(line):
    len_l = len(line)
    line = line.replace("  ", " ")
    while len_l > len(line):
        len_l = len(line)
        line = line.replace("  ", " ")
    return line


def read_dataFile(file_path, path_to_proj):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # удаляем первую строку
    lines.pop(0)
    # удаляем последние 14 строк (с запасом, чтобы не было мусора)
    lines = lines[:-14]
    # удаляем пробелы в начале и конце каждой строки
    lines = [line.strip() + "\n" for line in lines]
    # чистим пробелы
    lines = list(map(clear_space, lines))

    with open(path_to_proj + 'fil.dat', 'w') as f:
        f.writelines(lines)

    # Загрузка всех столбцов из файла
    data = pd.read_table(path_to_proj + "fil.dat", sep=" ", names=["t"] + ["ch{}".format(i) for i in range(1, 30)])

    os.remove(path_to_proj + 'fil.dat')

    return data.dropna(axis=1)


def save_df_toFile(df, name_csv, path_to_csv=""):
    if os.path.exists(path_to_csv) and os.path.exists(path_to_csv + name_csv):
        df.to_csv(path_to_csv + name_csv, mode='a', header=False, index=False)
    else:
        if not os.path.exists(path_to_csv):
            os.mkdir(path_to_csv)
        df.to_csv(path_to_csv + name_csv, index=False)


def save_results_toFiles(predictions, fragments, file_data_csv_name, meta_data, path_to_csv="",
                         edge=0.5, signal_maxLength=512, f_save_all=True, f_disp=False):
    """
    :param predictions:
    :param fragments:
    :param file_data_csv_name:
    :param meta_data: {"id": FILE_D_ID, "ch": selected_channel, "rate": SIGNAL_RATE}
    :param path_to_csv: default=""
    :param edge: default=0.5
    :param signal_maxLength: default=512
    :param f_disp:
    :param f_save_all:
    :return:
    """
    df = pd.DataFrame(columns=(['D_ID', 'Ch', 'Left', 'Right', 'Y', 'Length', 'Rate'] + [str(i) for i in range(signal_maxLength)]))

    if f_disp:
        print("|", end="")

    fragments_count = 0
    filaments_count = 0
    tot_filaments_mark = 0
    noise_count = 0

    iter_step = predictions.shape[0] // 10
    iter_count = 1

    pred_index = 0
    for i in range(predictions.shape[0]):
        fragment_x = fragments[0][i].tolist()
        fragment_values = fragments[1][i].tolist()

        # получение границ фрагмента из введённой строки
        fragment_range = [min(fragment_x), max(fragment_x)]
        # нормировка границ (если выходит за рамки сигнала) и получение его длительности
        fragment_range = [fragment_range[0], fragment_range[1]]
        fragment_length = fragment_range[1] - fragment_range[0]

        if fragment_length <= 0:
            print(f"===== Length error (len: {fragment_length}) =====")
            continue

        if len(fragment_x) <= 5 or len(fragment_x) >= 500:
            print("===== Warning =====")
            # запрос команды подтверждения
            print(f"Фрагмент {i} содержит меньше 5 или больше 500 точек ({len(fragment_x)} точек)")
            continue

        # получение с помощью интерполяции квадратичным сплайном нужного количества точек фрагмента (510 точек)
        fragment_interpolate_values = sc_i.interp1d(fragment_x, fragment_values, kind="quadratic")(
            np.linspace(fragment_x[0], fragment_x[-1], signal_maxLength))

        # получение метки фрагмента из прогнозированных данных
        if len(predictions.shape) > 1:
            fragment_mark = round(predictions[pred_index][0], 2)
        else:
            fragment_mark = round(predictions[pred_index], 2)

        pred_index += 1

        if f_save_all or fragment_mark >= edge:
            fragments_count += 1

            # добавление данных в Data Frame
            df.loc[-1] = [meta_data["id"], meta_data["ch"], min(fragment_range), max(fragment_range), fragment_mark, fragment_length, meta_data["rate"]] + list(
                fragment_interpolate_values)  # adding a row
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index

        # обработка автоматического сохранения Data Frame
        if fragment_mark < edge:
            noise_count += 1
        else:
            filaments_count += 1
            tot_filaments_mark += fragment_mark
        if fragments_count % 10 == 0:
            # print(f"Количество сохранённых фрагментов: {fragments_count}\n" +
            #       f"Филаментов: {filaments_count} (средняя оценка филаментов: {round(tot_filaments_mark / (filaments_count+1), 2)})" +
            #       f"\nНе филаментов: {noise_count}")

            # сохранение Data Frame
            save_df_toFile(df, file_data_csv_name, path_to_csv)

            # очистка Data Frame
            df = df.iloc[0:0]

        if f_disp and iter_step * iter_count <= i:
            iter_count += 1
            print(".", end="")

    # сохранение Data Frame
    if len(df.count(axis="rows")) > 0:
        save_df_toFile(df, file_data_csv_name, path_to_csv)

    if f_disp:
        print("|")

    print(f"Количество сохранённых фрагментов: {fragments_count}\n" +
          f"Филаментов: {filaments_count} (средняя оценка филаментов: {round(tot_filaments_mark / (filaments_count + 1), 2)})" +
          f"\nНе филаментов: {noise_count}")
