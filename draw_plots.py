import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

line_kwargs = {}#"drawstyle": "steps-post"

fig, ax = plt.subplots()
xlabel = "Recall"
ylabel = "Precision"
ax.set(xlabel=xlabel, ylabel=ylabel)

line_array = []
names = []

def plot_data(ax_p, path, name):

    data_plot2 = np.load(path)
    precision = np.around(data_plot2[:, 0], decimals=3)
    recall = np.around(data_plot2[:, 1], decimals=3)
    line, = ax.plot(recall, precision, **line_kwargs)
    line_array.append(line)
    names.append(name)

def gather_data(path, prec_arr, rec_arr):
    data_plot2 = np.load(path)
    prec_arr.append(np.around(data_plot2[:, 0], decimals=3))
    rec_arr.append(np.around(data_plot2[:, 1], decimals=3))
    return np.around(data_plot2[:, 4], decimals=3)

def get_diff(mean_line, line):
    return np.sum(mean_line - line)

def get_interpolation_line(line_x, line_y):
    new_line_x = np.linspace(0, 1, 1000)
    new_line_y = np.interp(new_line_x, np.flip(line_x), np.flip(line_y))
    #tck = interpolate.splrep(np.flip(line_x), np.flip(line_y))
    #new_line_y = interpolate.splev(new_line_x, tck)

    return new_line_x, new_line_y

def get_bounds(lines_y):
    lines_y_max = np.max(lines_y, axis=0)
    lines_y_min = np.min(lines_y, axis=0)
    return lines_y_min, lines_y_max





prec_arr, rec_arr = [], []
for i in range(9):
    fn = gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f1+100/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("f1+v2+100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)








prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f2+resNetInitFalse+v2+100/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("f2+v2+100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)








prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f5+resnetInitFalse+v2+100/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("f5+v2+100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)








prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-master/plots/original/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("focal+100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)






prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-master/plots/centerNetHead/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("focal + centerNet + 100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)








prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-master/plots/newFeatured/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_rec_arr.append(inter_line_x)
    inter_prec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
line_array.append(line)
names.append("focal + centerNet + barcode features + 100 mean")

plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)




ax.legend(line_array, names)
plt.show()

