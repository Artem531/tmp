import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import CubicSpline

line_kwargs = {}#"drawstyle": "steps-post"

fig, ax = plt.subplots()
xlabel = "th"
ylabel = "f5"
ax.set(xlabel=xlabel, ylabel=ylabel)

line_array = []
names = []

max_distance = []

def plot_data(ax_p, path, name):

    data_plot2 = np.load(path)
    precision = np.around(data_plot2[:, 0], decimals=3)
    recall = np.around(data_plot2[:, 1], decimals=3)
    line, = ax.plot(recall, precision, **line_kwargs)
    line_array.append(line)
    names.append(name)


# Граница минимального рекол в результатаз инференса (ниже нее нет пресижин)
min_rec_infer = 0
beta = 1
def gather_data(path, fn_arr, th_arr):
    global min_rec_infer
    data_plot2 = np.load(path)

    tp = data_plot2[:, 0][1:]
    fp = data_plot2[:, 1][1:]
    fn = data_plot2[:, 2][1:]
    th = data_plot2[:, 3][1:]

    print(fn.shape, th.shape)
    fn_arr.append(tp / (tp + (fp + beta**2 * fn) / (beta**2 + 1) ))
    th_arr.append(th)

base_path = ""
plots_num = 5
def gather_th(path):
    prec, th = [], []
    for i in range(plots_num):
        data_plot2 = np.load(base_path + path + f"/{str(i)}.npy")

        prec.append(np.around(data_plot2[:, 0], decimals=3))
        th.append(np.around(data_plot2[:, 5], decimals=3))
    return np.array(prec), np.array(th)


def get_diff(mean_line, line):
    return np.sum(mean_line - line)

def get_bounds(lines_y):
    lines_y_max = np.max(lines_y, axis=0)
    lines_y_min = np.min(lines_y, axis=0)
    return lines_y_min, lines_y_max

def plot_data(name, display_name):
    fn_arr, th_arr = [], []

    len_fn = 0
    for i in range(plots_num):
        gather_data(base_path + name + f"/{str(i)}.npy", fn_arr, th_arr)

    fn_arr = np.array(fn_arr)
    th_arr = np.array(th_arr)

    fn_mean = np.median(fn_arr, axis=0)
    th_mean = np.mean(th_arr, axis=0)

    line, = ax.plot(th_mean, fn_mean, **line_kwargs)
    line_array.append(line)

    names.append(display_name)

    min_fn_line_bound, max_fn_line_bound = get_bounds(fn_arr)

    #plt.fill_between(th_mean, min_fn_line_bound, max_fn_line_bound, alpha=0.4)


name = "/home/artem/PycharmProjects/backboned-unet-new/plots/original+focal+100"
display_name = "CenterNet"
plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_infer/f1+100"
# display_name = "Многоцелевой f1 loss 100 эпох"
# plot_data(name, display_name)
#
# #
# name = "real_data/iou/iou_real_infer/f1+v1+100"
# display_name = "Dice loss/Не многоцелевой f1 loss"
# plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_syn_infer/f5+100"
# display_name = "Многоцелевой f5 loss 100 эпох"
# plot_data(name, display_name)
# #

# name = "real_data/iou/iou_real_syn_infer/newFeaturesPretrain"
# display_name = "CenterNet + качественные признаки"
# plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_syn_infer/original+focal+100"
# display_name = "Focal loss 100 эпох"
# plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_syn_infer/original+focal+200"
# display_name = "Focal loss 200 эпох"
# plot_data(name, display_name)
#
# #
# name = "real_data/iou/iou_real_infer/PatternsModels+FocalLoss"
# display_name = "Focal loss 100 эпох Multilable"
# plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_infer/PatternsModels+FocalLoss+200"
# display_name = "Focal loss 200 эпох Multilable"
# plot_data(name, display_name)
#
#
# name = "real_data/iou/iou_real_infer/PatternsMulticlass+focal"
# display_name = "Focal loss 100 эпох Multiclass"
# plot_data(name, display_name)
#

mean_fn = []


print(max_distance)
ax.legend(line_array, names)
plt.show()

