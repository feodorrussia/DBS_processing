import numpy as np
from numba import jit
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


def shred_param_calc(frag_points, shred_points, COVER_PERCENT=0.5):
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


def length_preproc(preproc_fragments, rate,
                   MIN_LENGTH_MCS=0.008,
                   MAX_LENGTH_MCS=0.03,
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
        if MAX_LENGTH_MCS >= frag_len * 1.5 and frag_len >= MIN_LENGTH_MCS:
            result_fragments.append(np.array(fragment))
        elif MAX_LENGTH_MCS < frag_len * 1.5:
            n_cuts, cut_step = shred_param_calc(len(fragment), points_frag(SHRED_LENGTH_MCS, rate))
            # print(n_cuts, cut_step, points_frag(SHRED_LENGTH_MCS, rate), len(fragment), n_cuts * cut_step,
            #       n_cuts * points_frag(SHRED_LENGTH_MCS, rate),
            #       abs(len(fragment) - n_cuts * points_frag(SHRED_LENGTH_MCS, rate)) / len(fragment))
            for i in range(n_cuts):
                result_fragments.append(np.array(fragment[i * cut_step:min((i + 1) * cut_step, len(fragment) - 1)]))
            if len(fragment[n_cuts * cut_step - 1:]) > 5:
                result_fragments.append(np.array(fragment[n_cuts * cut_step - 1:]))

    # print(len(result_fragments))
    # print([i.shape for i in result_fragments])
    # i = input()

    return np.array(result_fragments, dtype="object")


@jit
def count_fft_max(arr: np.array, arr2: np.array = None):
    f_increase = False
    max_count = 0
    max_v = 0
    maximums = []

    for i in range(1, arr.shape[0]):
        if not f_increase and arr[i - 1] < arr[i]:
            f_increase = True
            max_v = arr[i]
            # print(arr[i - 1], arr[i])
        if f_increase:
            max_v = max(arr[i], max_v)
            if arr[i - 1] > arr[i] and abs(max_v - arr[i]) / max_v > 0.3:
                f_increase = False
                # print(max_v, arr[i], abs(max_v - arr[i]) / max_v, "\n")
                maximums.append(max_v)
                max_count += 1
    if len(maximums) > 0:
        if arr2 is not None:
            max_max_freq = arr2[arr == max(maximums)][0]
        else:
            max_max_freq = -1.0
        if len(maximums) > 1:
            maximums = np.array(maximums)
            mean_max_ratio = (maximums[maximums != maximums.max()] / maximums.max()).max()
        else:
            mean_max_ratio = 0.0
    else:
        return [0.0, 0.0, -1.0]

    return [max_count, mean_max_ratio, max_max_freq]


@jit
def down_to_zero(x, edge=0.05):
    return x if x > edge else 0.0


@jit
def periods_count(fragment):
    k = 0
    mean = np.mean(fragment)
    for j in range(len(fragment) - 1):
        if (fragment[j] - mean) * (fragment[j + 1] - mean) < 0:
            k += 1
    return k


def fft_butter_skewness_filtering(t_data, signal_data, rate, f_disp=False):
    """
    :param t_data:  np.array(), main signal t_data
    :param signal_data: list(np.array()), channels list of signal y_data
    :param rate: int, signal rate (4 / 10)
    :param log_df: data frame for plotting app (testing)
    :return: [x_lists, y_lists], filtered fragments
    :param f_disp:
    """
    # CONSTANTS
    TOLERANCE = 1  # Чем выше, тем больше шанс получить два филамента на одной картинке
    MIN_PERIODS = 3  # В среднем количество колебаний на графике, начальный порог
    MAX_PERIODS = 50  #
    MAX_FFT_MAX = 10  #
    MAX_SKEWNESS = 0.4  # Абсолютная асимметрия
    MAX_RATIO_FFT = 0.5
    BOARDERS_PERCENT = 0.5  # Сколько процентов длины добавляем слева и справа от филамента.
    MIN_LENGTH_MCS = 0.008
    REGION_LENGTH_MCS = 0.012
    MAX_LENGTH_MCS = 0.035

    CHANNELS_N = len(signal_data)

    region = (int(REGION_LENGTH_MCS * rate * 1000) + 1) // 2  # получаем количество точек для рассматриваемого "окна"

    if f_disp:
        print("|", end="")

    n_diff = 1
    signal_data_d1 = [np.diff(signal_data[i], n=n_diff) for i in range(CHANNELS_N)]
    # if log_df is not None:
    #     log_df[f"ch11_{n_diff}d"] = np.concatenate([np.diff(signal_data, n=n_diff), [0]*n_diff])

    b, a = signal.butter(5, 0.1)

    signal_data_d1 = [signal.filtfilt(b, a, signal_data_d1[i]) for i in range(CHANNELS_N)]

    preprocessed_ind_data = [[]]
    f_fragment = False

    search_step = region // 2
    iter_step = signal_data.shape[0] // 10
    iter_count = 1
    point = region
    while point < signal_data.shape[0] - region - n_diff:
        # Выбираем в качестве аномалий то, что +- стандартное отклонение. В лоб, но может сработать
        f_point = False
        for ch_i in range(CHANNELS_N):
            fragment_d1 = signal_data_d1[ch_i][point - region:point + region]

            periods_d1 = periods_count(fragment_d1)
            periods = periods_count(signal_data[point - region:point + region])

            fft = np.fft.fft(fragment_d1)
            fft_v = fft.real ** 2 + fft.imag ** 2
            filter_values = np.vectorize(lambda x: down_to_zero(x, edge=fft_v.max() * 0.05))
            fft_v_filter = filter_values(fft_v)

            frequency = np.unique(np.abs(np.fft.fftfreq(fragment_d1.shape[0])))
            frequency = frequency[(frequency >= 0.0) & (frequency <= 0.1)]
            fft_v_filter = fft_v_filter[:frequency.shape[0]]

            max_fft, max_ratio_fft, fr_max_fft = count_fft_max(fft_v_filter, frequency)

            abs_skewness = np.abs(skew(fragment_d1))

            if 0 < max_fft <= MAX_FFT_MAX and periods_d1 >= MIN_PERIODS and periods <= MAX_PERIODS and max_ratio_fft < MAX_RATIO_FFT and abs_skewness < MAX_SKEWNESS:
                f_point = True

        if f_point:
            f_fragment = True
            preprocessed_ind_data[-1].append(point)
            search_step = 1
        else:
            search_step = region // 2

            if f_fragment:
                preprocessed_ind_data.append([])
            f_fragment = False

        point += search_step
        if f_disp and iter_step * iter_count <= point:
            iter_count += 1
            print(".", end="")

    preprocessed_ind_data = length_preproc(preprocessed_ind_data, rate, SHRED_LENGTH_MCS=REGION_LENGTH_MCS * 1.2)
    fragments = [[]] + [[] for i in range(CHANNELS_N)]

    iter_step = len(preprocessed_ind_data) // 5
    iter_count = 1
    for i in range(len(preprocessed_ind_data)):
        fragment_ind = preprocessed_ind_data[i]
        fragment_len = fragment_ind.shape[0]

        l_edge = max(fragment_ind[0] - int(BOARDERS_PERCENT * fragment_len), 0)
        r_edge = min(fragment_ind[fragment_len - 1] + int(BOARDERS_PERCENT * fragment_len), t_data.shape[0] - 1)

        fragments[0].append(t_data[l_edge:r_edge + 1])
        for ch_i in range(CHANNELS_N):
            fragments[ch_i + 1].append(signal_data[ch_i][l_edge:r_edge + 1])

        if f_disp and iter_step * iter_count <= i:
            iter_count += 1
            print(".", end="")

    if f_disp:
        print("|")

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
            continue

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
