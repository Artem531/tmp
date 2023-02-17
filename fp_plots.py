import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import CubicSpline

line_kwargs = {}#"drawstyle": "steps-post"

fig, ax = plt.subplots()
xlabel = "th"
ylabel = "fp"
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

def gather_data(path, fn_arr, th_arr):
    global min_rec_infer
    data_plot2 = np.load(path)

    tp = data_plot2[:, 0][10:-1]
    fp = data_plot2[:, 1][10:-1]
    fn = data_plot2[:, 2][10:-1]
    th = data_plot2[:, 3][10:-1]

    print(fn.shape, th.shape)
    fn_arr.append(fp)
    th_arr.append(th)

base_path = "/home/artem/PycharmProjects/backboned-unet-new/plots/"
plots_num = 9
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

    fn_mean = np.mean(fn_arr, axis=0)
    th_mean = np.mean(th_arr, axis=0)

    line, = ax.plot(th_mean, fn_mean, **line_kwargs)
    line_array.append(line)

    names.append(display_name)

    min_fn_line_bound, max_fn_line_bound = get_bounds(fn_arr)

    plt.fill_between(th_mean, min_fn_line_bound, max_fn_line_bound, alpha=0.4)

# name = "f1+100+syntetic"
# display_name = "f1 версия 1"
# plot_data(name, display_name)
#
name = "Original+200+syntetic+iou"
display_name = "Original syntetic"
plot_data(name, display_name)

name = "original+200+syntetic+iou+ivan"
display_name = "original+200+syntetic+iou+ivan"
plot_data(name, display_name)

# name = "resnetCenterNetUnet+100+syntetic"
# display_name = "resnetCenterNetUnet syntetic"
# plot_data(name, display_name)

#
# name = "testNewFeatures+100+syntetic+iou"
# display_name = "testNewFeatures+100+syntetic"
# plot_data(name, display_name)

# name = "resnetCenterNetUnet+100+syntetic"
# display_name = "resnetCenterNetUnet"
# plot_data(name, display_name)
#
#
# name = "f1+100+syntetic"
# display_name = "f1+100+syntetic"
# plot_data(name, display_name)
#
# name = "f5+100+syntetic"
# display_name = "f5+100+syntetic"
# plot_data(name, display_name)
#
#
# #
# name = "Original+200+syntetic"
# display_name = "Original+200+syntetic"
# plot_data(name, display_name)


# name = "f5+100+syntetic+iou"
# display_name = "f5+100+syntetic"
# plot_data(name, display_name)
#
# name = "Original+200+syntetic+iou"
# display_name = "Original syntetic"
# plot_data(name, display_name)

mean_fn = []


print(max_distance)
ax.legend(line_array, names)
plt.show()

