import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


def focal_crossentropy(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    alpha_factor = 1
    modulating_factor = 1

    alpha_factor = y_true * 0.25 + ((1 - 0.25) * (1 - y_true))
    modulating_factor = K.pow((1 - p_t), 2.0)

    # compute the final loss and return
    return K.mean(alpha_factor * modulating_factor * bce, axis=-1)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    bce = K.binary_crossentropy(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    modulating_factor = K.pow((1 - p_t), gamma)

    # compute the final loss and return
    return K.mean(alpha_factor * modulating_factor * bce, axis=-1)


def focal_loss_01(y_true, y_pred, alpha=0.1, gamma=2.0):
    bce = K.binary_crossentropy(y_true, y_pred)

    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

    alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
    modulating_factor = K.pow((1 - p_t), gamma)

    # compute the final loss and return
    return K.mean(alpha_factor * modulating_factor * bce, axis=-1)


# Metrics function
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f_m(y_true, y_pred, b=1):
    precision = precision_m(y_true, y_pred)  # y_true
    recall = recall_m(y_true, y_pred)  # y_pred
    return (1 + b ** 2) * ((precision * recall) / (b ** 2 * precision + recall + K.epsilon()))


def class_scores_processing(y_pred, f_over=False, edge=0.9):
    if f_over:
        return 1. if y_pred[0] <= y_pred[1] or y_pred[1] >= edge else 0.
    else:
        return y_pred[1] if y_pred[0] <= y_pred[1] or y_pred[1] >= edge else 0.


# Построение графика Количества отобраннных филаментов от величины граничной вероятности
def plot_predictionCurve(pred_y, test_y=None):
    edges = np.linspace(0, 1, 100)
    fil_nums = []
    for i in edges:
        filtered = pred_y >= i
        fil_nums.append(len(list(filter(lambda x: x, filtered))))

    fig, ax = plt.subplots()
    ax.hist(pred_y)
    if test_y is not None:
        ax.hist(test_y)
    ax.grid()
    ax.set_ylabel('Numbers of filtered filaments')
    ax.set_xlabel('Edge')
    ax.set_title('Filtered filaments by edge')
    plt.show()
    plt.clf()
