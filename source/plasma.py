import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import skew
from scipy import signal
import joblib
import keras
import tensorflow
import os
import shutil
import gc
import sklearn
import scipy.interpolate as sc_i


def x_in_y(query, base):
    """
    The function returns the index of the subsequence in the sequence

    query: list - subsequence
    base: list - sequence
    """
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i + l] == query:
            return i
    return False


path_to_proj = "DBS-processing/"  # DBS-processing/
path_to_csv = ""  # data_csv/

if not os.path.exists(path_to_csv):
    os.mkdir(path_to_csv)

file_path = input("Введите имя файла. Доступные файлы:\n" + "\n".join(
    list(filter(lambda x: '.dat' in x or '.txt' in x, os.listdir(path_to_csv)))) + "\n")
file_name = file_path.split('/')[-1]

file_fragments_csv_name = file_path[:-4] + "_fragments.csv"

with open(file_path, 'r') as f:
    lines = f.readlines()

# удаляем первую строку
lines.pop(0)

# удаляем последние 4 строки
lines = lines[:-4]

# удаляем 4 пробела в начале каждой строки
lines = [line[4:] if line.startswith('    ') else line for line in lines]

# заменяем двойные пробелы на одинарные
lines = [line.replace('  ', ' ') for line in lines]

with open(path_to_proj + 'fil.dat', 'w') as f:
    f.writelines(lines)

with open(path_to_proj + 'fil.dat', 'r') as f:
    lines = f.readlines()

# удаляем первую строку
lines.pop(0)

# удаляем последние 4 строки
lines = lines[:-4]

# удаляем 4 пробела в начале каждой строки
lines = [line[4:] if line.startswith('    ') else line for line in lines]

# заменяем двойные пробелы на одинарные
lines = [line.replace('  ', ' ') for line in lines]

with open(path_to_proj + 'fil.dat', 'w') as f:
    f.writelines(lines)

gc.collect()

# Загрузка всех столбцов из файла
data = pd.read_table(path_to_proj + "fil.dat", sep=" ", names=["t"] + ["ch{}".format(i) for i in range(1, 300)])

print(f"Выбран файл {file_path}")
# Предложение пользователю выбрать канал
available_channels = [col for col in data.columns if col != "t" and str(data[col][0]) != 'nan']
print("Доступные каналы:", ', '.join(available_channels))
selected_channel = input("Выберите канал: ")

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

y_d2 = np.diff(y, n=2)
x_d2 = x[:-2]

gc.collect()

b, a = signal.butter(3, 0.4)

y_d2 = signal.filtfilt(b, a, y_d2)

# Просто выбираем в качестве аномалий то, что +- стандартное отклонение. В лоб, но может сработать

m = y_d2.mean()
std = y_d2.std()

# преобразование Фурье и получение массива частот
fft = np.fft.fft(y_d2)
frequency = np.abs(np.fft.fftfreq(len(y_d2)))

colors = ["blue", "red"]
region = 3

y_d2_logic = np.zeros(len(y_d2))

for i in range(3, len(y_d2_logic) - 3):
    condition = False
    for k in range(-region, region + 1):
        condition = condition or ((y_d2[i + k] > m + std) or (y_d2[i + k] < m - std)) \
                    and frequency[i] > 0.03
    if condition:
        y_d2_logic[i] = True

y_f = np.array([y[i] if y_d2_logic[i] else 0 for i in range(len(y_d2_logic))])

tolerance = 1  # Чем выше, тем больше шанс получить два филамента на одной картинке
periods = 3  # В среднем количество колебаний на графике, начальный порог
length = 10  # Характеристика длины филамента
sinusoidality = 0.8  # Абсолютная асимметрия
edges = 20  # Сколько точек добавляем слева и справа от филамента.

preprocessed = ";".join(map(str, y_f)).split("0.0;" * tolerance)
preprocessed = [i.split(";") for i in preprocessed if len(i) > 1]
for i in range(len(preprocessed)):
    preprocessed[i] = [float(j) for j in preprocessed[i] if j != ""]
final = np.array([i for i in preprocessed if len(i) > length], dtype="object")

for i in range(len(final)):
    final[i] = np.array(final[i])

filaments = [[], []]
filaments_smooth = [[], []]

for i in range(len(final)):
    y_ = final[i]

    k = 0
    mean = np.mean(y_)
    for j in range(len(y_) - 1):
        if (y_[j] - mean) * (y_[j + 1] - mean) < 0:
            k += 1

    abs_skewness = np.abs(skew(y_))

    if k > periods * 2 and abs_skewness < sinusoidality:
        r = x_in_y(final[i][:5].tolist(), data[selected_channel].tolist())
        x_id = np.array(list(range(r - edges, r + edges + final[i].shape[0])))

        x_ = data.t[x_id]
        y_ = data[selected_channel][x_id]
        filaments[0].append(x_)
        filaments[1].append(y_)

        x_smooth = np.linspace(x_.min(), x_.max(), 64)
        y_smooth = interpolate.make_interp_spline(x_, y_)(x_smooth)
        filaments_smooth[0].append(x_smooth)
        filaments_smooth[1].append(y_smooth)

gc.collect()

neuro_filter = keras.models.load_model(path_to_proj + "models/neuro_filter.h5")
scaler = joblib.load(path_to_proj + "models/scaler_for_neuro_filter.pkl")

predictions = neuro_filter.predict(scaler.transform(filaments_smooth[1]))

# Построение графика Количества отобраннных филаментов от величины граничной вероятности
# edges = np.linspace(0, 1, 100)
# fil_nums = []
# for i in edges:
#     filtered = predictions > i
#     fil_nums.append(len(list(filter(lambda x: x, filtered))))
#
# fig, ax = plt.subplots()
# ax.plot(edges, fil_nums)
# ax.grid()
# ax.set_ylabel('Numbers of filtered filaments')
# ax.set_title('Filtered filaments by edge')
# plt.show()
# plt.clf()
# gc.collect()

edge = 0.9
filtered = predictions > edge

print("==========================================")
print(f"Количество найденных филаментов: {len(filaments[0])}")
print("==========================================")
print(f"Количество отобранных филаментов: {len(list(filter(lambda x: x, filtered)))}")
print("==========================================")

gc.collect()

if not os.path.exists(path_to_csv):
    os.mkdir(path_to_csv)
if not os.path.exists(path_to_csv + "result_data/"):
    os.mkdir(path_to_csv + "result_data/")

name_csv = f"result_data/{file_path[:-4]}_result_data.csv"
file_fragments_csv_name = f"result_fragments/{file_path[:-4]}_result_fragments.csv"

signal_maxLength = 512
FILE_D_ID = file_path[:5]  # "00000"
SIGNAL_RATE = 4


df = pd.DataFrame(columns=(['D_ID', 'Ch', 'Y', 'Length', 'Rate'] + [str(i) for i in range(signal_maxLength)]))
df_2 = pd.DataFrame(columns=(['Ch', 'Y', 'Left', 'Right', 'Rate']))

#
fragments_count = 0
filaments_count = 0
tot_filaments_mark = 0
noise_count = 0

for i in range(len(filaments[0])):
    fragment_x = filaments[0][i].to_list()
    fragment_values = filaments[1][i].to_list()

    # получение границ фрагмента из введённой строки
    fragment_range = [min(fragment_x), max(fragment_x)]
    # нормировка границ (если выходит за рамки сигнала) и получение его длительности
    fragment_range = [max(fragment_range[0], start), min(fragment_range[1], end)]
    fragment_length = fragment_range[1] - fragment_range[0]

    if fragment_length <= 0:
        print("===== Length error =====")
        continue

    if len(fragment_x) <= 5 or len(fragment_x) >= 500:
        print("===== Warning =====")
        # запрос команды подтверждения
        print(f"Фрагмент {i} содержит меньше 5 или больше 500 точек ({len(fragment_x)} точек)")

    # получение с помощью интерполяции квадратичным сплайном нужного количества точек фрагмента (510 точек)
    fragment_interpolate_values = sc_i.interp1d(fragment_x, fragment_values, kind="quadratic")(
        np.linspace(fragment_x[0], fragment_x[-1], signal_maxLength))

    # получение метки фрагмента: 1 - филамент, 0 - не филамент
    fragment_mark = round(predictions[i][0], 2)

    # добавление данных в Data Frame
    df.loc[-1] = [FILE_D_ID, selected_channel, fragment_mark, fragment_length, SIGNAL_RATE] + list(
        fragment_interpolate_values)  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index

    # добавление данных в Data Frame 2
    df_2.loc[-1] = [selected_channel, fragment_mark, min(fragment_range), max(fragment_range),
                    SIGNAL_RATE]  # adding a row
    df_2.index = df_2.index + 1  # shifting index
    df_2 = df_2.sort_index()  # sorting by index

    # обработка автоматического сохранения Data Frame
    fragments_count += 1
    if fragment_mark < edge:
        noise_count += 1
    else:
        filaments_count += 1
        tot_filaments_mark += fragment_mark
    if fragments_count % 10 == 0:
        print(f"Количество сохранённых фрагментов: {fragments_count}\n" +
              f"Филаментов: {filaments_count} (средняя оценка филаментов: {round(tot_filaments_mark / filaments_count, 2)})" +
              f"\nНе филаментов: {noise_count}")

        if os.path.exists(path_to_csv) and os.path.exists(path_to_csv + name_csv):
            df.to_csv(path_to_csv + name_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(path_to_csv + name_csv, index=False)
        # очистка Data Frame
        df = df.iloc[0:0]

        if os.path.exists(path_to_csv) and os.path.exists(path_to_csv + file_fragments_csv_name):
            df_2.to_csv(path_to_csv + file_fragments_csv_name, mode='a', header=False, index=False)
        else:
            df_2.to_csv(path_to_csv + file_fragments_csv_name, index=False)
        # очистка Data Frame
        df_2 = df_2.iloc[0:0]

# сохранение Data Frame
if len(df.count(axis="rows")) > 0:
    if os.path.exists(path_to_csv) and os.path.exists(path_to_csv + name_csv):
        df.to_csv(path_to_csv + name_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(path_to_csv + name_csv, index=False)

    if os.path.exists(path_to_csv) and os.path.exists(path_to_csv + file_fragments_csv_name):
        df_2.to_csv(path_to_csv + file_fragments_csv_name, mode='a', header=False, index=False)
    else:
        df_2.to_csv(path_to_csv + file_fragments_csv_name, index=False)

print(f"Количество сохранённых фрагментов: {fragments_count}\n" +
              f"Филаментов: {filaments_count} (средняя оценка филаментов: {round(tot_filaments_mark / filaments_count, 2)})" +
      f"\nНе филаментов: {noise_count}")

os.remove(path_to_proj + 'fil.dat')
gc.collect()
