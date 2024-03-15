import numpy as np
from scipy import signal, interpolate
from scipy.stats import skew


def x_in_y(query, base):
    """
    The function returns the index of the subsequence in the sequence
    :param query: list - subsequence
    :param base: list - sequence
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


def length_frag(n_points, rate):
    """
    :param n_points: int, num of points in fragment
    :param rate: int, signal rate (4 / 10)
    :return: float, length of fragment in mcs
    """
    return n_points / (rate * 1000)


def points_frag(length, rate):
    """
    :param length: float, length of fragment
    :param rate: int, signal rate (4 / 10)
    :return: float, length of fragment in mcs
    """
    return int(length * 1000 * rate)


def shred_param_calc(frag_points, shred_points, COVER_PERCENT=0.7):
    """
    Function for calculating shredding parameter
    :param frag_points: num points of shredding fragment
    :param shred_points: num points of result fragments
    :return: cut_num, cut_offset: int, int; num of cuts & offset of cuts
    :param COVER_PERCENT: =0.7, how much slicing cover fragment
    """
    cut_num = frag_points // shred_points + 1
    while abs(frag_points - shred_points * cut_num) / frag_points < COVER_PERCENT:
        cut_num += 1
    cut_offset = frag_points // cut_num

    return cut_num, cut_offset


def lenght_preproc(preproc_fragments, rate,
                   MIN_LENGTH_MCS=0.008,
                   MAX_LENGTH_MCS=0.035,
                   SHRED_LENGTH_MCS=0.025):
    """
    Function for filtering too short fragments & shredding too long
    :param preproc_fragments: list(list(float)), main signal y_data
    :param rate: int, signal rate (4 / 10)
    :param MAX_LENGTH_MCS: =0.008 mcs, < delete
    :param MIN_LENGTH_MCS: =0.035 mcs, > shred
    :param SHRED_LENGTH_MCS: =0.025 mcs, length of shredding fragments
    :return: np.array(np.array(float))
    """
    result_fragments = []

    for fragment in preproc_fragments:
        frag_len = length_frag(len(fragment), rate)
        if MAX_LENGTH_MCS >= frag_len >= MIN_LENGTH_MCS:
            result_fragments.append(np.array(fragment))
        elif MAX_LENGTH_MCS <= frag_len * 1.5:
            n_cuts, cut_step = shred_param_calc(len(fragment), points_frag(SHRED_LENGTH_MCS, rate))
            # print(n_cuts, cut_step, points_frag(SHRED_LENGTH_MCS, rate), len(fragment), n_cuts * cut_step,
            #       n_cuts * points_frag(SHRED_LENGTH_MCS, rate),
            #       abs(len(fragment) - n_cuts * points_frag(SHRED_LENGTH_MCS, rate)) / len(fragment))
            for i in range(n_cuts):
                result_fragments.append(np.array(fragment[i * cut_step:min((i + 1) * cut_step, len(fragment) - 1)]))
            if abs(n_cuts * cut_step - len(fragment)) >= points_frag(SHRED_LENGTH_MCS, rate) / 2:
                result_fragments.append(np.array(fragment[n_cuts * cut_step - 1:]))

    # print(len(result_fragments))
    # print([i.shape for i in result_fragments])
    # i = input()

    return np.array(result_fragments, dtype="object")


def fft_butter_skewness_filtering(x_data, signal_data, rate=4):
    """
    :param x_data:  np.array(), main signal x_data
    :param signal_data: np.array(), main signal y_data
    :param rate: int, signal rate (4 / 10)
    :return: [x_lists, y_lists], filtered fragments
    """
    # CONSTANTS
    TOLERANCE = 1  # Чем выше, тем больше шанс получить два филамента на одной картинке
    MIN_PERIODS = 4  # В среднем количество колебаний на графике, начальный порог
    SINUSOIDALITY = 0.7  # Абсолютная асимметрия
    BOARDERS_PERCENT = 0.5  # Сколько процентов длины добавляем слева и справа от филамента.
    MIN_LENGTH_MCS = 0.008
    MAX_LENGTH_MCS = 0.035

    signal_data_d2 = np.diff(signal_data, n=2)

    b, a = signal.butter(3, 0.4)

    signal_data_d2 = signal.filtfilt(b, a, signal_data_d2)

    # Просто выбираем в качестве аномалий то, что +- стандартное отклонение. В лоб, но может сработать
    m = signal_data_d2.mean()
    std = signal_data_d2.std()

    # преобразование Фурье и получение массива частот
    fft = np.fft.fft(signal_data_d2)
    frequency = np.abs(np.fft.fftfreq(len(signal_data_d2)))

    region = 3

    y_d2_logic = np.zeros(len(signal_data_d2))

    for i in range(3, len(y_d2_logic) - 3):
        condition = False
        for k in range(-region, region + 1):
            condition = condition or ((signal_data_d2[i + k] > m + std) or (signal_data_d2[i + k] < m - std)) \
                        and frequency[i] > 0.03
        if condition:
            y_d2_logic[i] = True

    signal_data_f = np.array([signal_data[i] if y_d2_logic[i] else 0 for i in range(len(y_d2_logic))])

    preprocessed = ";".join(map(str, signal_data_f)).split("0.0;" * TOLERANCE)
    preprocessed = [i.split(";") for i in preprocessed if len(i) > 1]
    for i in range(len(preprocessed)):
        # print(preprocessed[i] if "-" in preprocessed[i] else "", end="")
        preprocessed[i] = [float(j) for j in preprocessed[i] if j != "" and j != "-"]

    filtered_data = lenght_preproc(preprocessed, rate)

    fragments = [[], []]

    for i in range(len(filtered_data)):
        y_ = filtered_data[i]

        k = 0
        mean = np.mean(y_)
        for j in range(len(y_) - 1):
            if (y_[j] - mean) * (y_[j + 1] - mean) < 0:
                k += 1

        abs_skewness = np.abs(skew(y_))

        if k > MIN_PERIODS * 2 and abs_skewness < SINUSOIDALITY:
            r = x_in_y(filtered_data[i][:5].tolist(), signal_data.tolist())
            x_id = np.array(
                list(range(r - int(BOARDERS_PERCENT * filtered_data[i].shape[0]),
                           r + int((BOARDERS_PERCENT + 1) * filtered_data[i].shape[0]))))

            fragments[0].append(x_data[x_id])
            fragments[1].append(signal_data[x_id])

    return fragments


def fragment_smoothing_preproc(x, y, SIGNAL_RATE, FRAGMENT_LEN):
    x_smooth = np.linspace(x.min(), x.max(), FRAGMENT_LEN)
    y_smooth = interpolate.make_interp_spline(x, y)(x_smooth)

    length_fragments = (x.max() - x.min()) * 100
    rate_fragments = SIGNAL_RATE / 10
    signal_fragments = y_smooth

    # get min & max points values from all check data
    max_point, min_point = signal_fragments.max(), signal_fragments.min()
    # normalise all values
    signal_fragments = (signal_fragments - (max_point + min_point) / 2) / (max_point - min_point)

    return [signal_fragments, [length_fragments, rate_fragments]]


def data_converting_CNN(fragments, rate=4, to_len=512):
    """
    Function for converting fragments for filtering with concatenated neural network
    :param fragments: [x_lists, y_lists], data of fragments
    :param rate: default=4, rate of the signal
    :param to_len: default=512, desire number of point (neurons on the 1st layer)
    :return:
    """

    fragments_signal = []
    fragments_meta = []

    for i in range(len(fragments[0])):
        if len(fragments[0][i]) <= 5 or len(fragments[0][i]) >= 500:
            print("===== Warning =====")
            # запрос команды подтверждения
            print(f"Фрагмент {i} содержит меньше 5 или больше 500 точек ({len(fragments[0][i])} точек)")

        norm_signal_fragment, norm_meta_data = fragment_smoothing_preproc(fragments[0][i], fragments[1][i],
                                                                          rate, to_len)

        # saving data with normalising for concatenated net
        fragments_signal.append(norm_signal_fragment)
        fragments_meta.append(norm_meta_data)
    return [np.array(fragments_signal), np.array(fragments_meta)]


def data_converting_OLD(fragments):
    """
    Function for converting fragments for OPD filter&scaler neural network
    :param fragments: [t_list, y_list], data of fragments
    :return: [x_lists, y_lists], smoothed fragments
    """
    filaments_smooth = [[], []]

    for i in range(len(fragments[0])):
        x_smooth = np.linspace(fragments[i][0].min(), fragments[i][0].max(), 64)
        y_smooth = interpolate.make_interp_spline(fragments[i][0], fragments[i][1])(x_smooth)

        filaments_smooth[0].append(x_smooth)
        filaments_smooth[1].append(y_smooth)
    return filaments_smooth
