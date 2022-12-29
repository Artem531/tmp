import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

line_kwargs = {"drawstyle": "steps-post"}

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
#
# #plot_data(ax, "/home/artem/PycharmProjects/backboned-unet-master/plots/exmp1.npy", "base")
# #plot_data(ax, "/home/artem/PycharmProjects/backboned-unet-master/plots/exmp1.npy", "base")
# prec_arr, rec_arr = [], []
# for i in range(0, 10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/Unet+resnetInitFalse/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
# print(prec_arr, rec_arr)
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("focal loss mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)
#
# ax.legend(line_array, names)
#
# line_kwargs = {"drawstyle": "steps-post"}
#
#
#








# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f1+resnetInitFalse+v2/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("f1 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)


# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/Unet+f1+resnetInitFalse/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("focal loss + f1 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)
#
#



# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f2+resnetInitFalse+v2/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("f2 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)


#
# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f05+resnetInitFalse+v2/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("f0.5 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)


# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f5+resnetInitFalse+v2/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("f5 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)

#
# prec_arr, rec_arr = [], []
# for i in range(10):
#     gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f5+resnetInitFalse+v1/{str(i)}.npy", prec_arr, rec_arr)
#
# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_prec_arr.append(inter_line_x)
#     inter_rec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
# #
# # i = 0
# # for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
# #     i += 1
# #     line, = ax.plot(line_x, line_y, **line_kwargs)
# #     line_array.append(line)
# #     names.append(str(i))
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
# line_array.append(line)
# names.append("f5 v1 mean")
#
# plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)





prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f5+resnetInitFalse+v2+30/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_prec_arr.append(inter_line_x)
    inter_rec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
#
# i = 0
# for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
#     i += 1
#     line, = ax.plot(line_x, line_y, **line_kwargs)
#     line_array.append(line)
#     names.append(str(i))

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
line_array.append(line)
names.append("f1+v2+30 mean")


plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)



prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f1+resnetInitFalse+v2+30/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_prec_arr.append(inter_line_x)
    inter_rec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
#
# i = 0
# for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
#     i += 1
#     line, = ax.plot(line_x, line_y, **line_kwargs)
#     line_array.append(line)
#     names.append(str(i))

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
line_array.append(line)
names.append("f5+v2+30 mean")

plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)


prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/f2+resnetInitFalse+v2+30/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_prec_arr.append(inter_line_x)
    inter_rec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
#
# i = 0
# for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
#     i += 1
#     line, = ax.plot(line_x, line_y, **line_kwargs)
#     line_array.append(line)
#     names.append(str(i))

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
line_array.append(line)
names.append("f2+v2+30 mean")

plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)






prec_arr, rec_arr = [], []
for i in range(10):
    gather_data(f"/home/artem/PycharmProjects/backboned-unet-new/plots/unet+focal+resnet+30/{str(i)}.npy", prec_arr, rec_arr)

prec_arr = np.array(prec_arr)
rec_arr = np.array(rec_arr)

inter_prec_arr = []
inter_rec_arr = []

for line_x, line_y in zip(rec_arr, prec_arr):
    inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)

    inter_prec_arr.append(inter_line_x)
    inter_rec_arr.append(inter_line_y)

inter_prec_arr = np.array(inter_prec_arr)
inter_rec_arr = np.array(inter_rec_arr)

min_line_bound, max_line_bound = get_bounds(inter_rec_arr)
#
# i = 0
# for line_x, line_y in zip(inter_prec_arr, inter_rec_arr):
#     i += 1
#     line, = ax.plot(line_x, line_y, **line_kwargs)
#     line_array.append(line)
#     names.append(str(i))

prec_mean = np.mean(inter_prec_arr, axis=0)
rec_mean = np.mean(inter_rec_arr, axis=0)

line, = ax.plot(prec_mean, rec_mean, **line_kwargs)
line_array.append(line)
names.append("focal loss 30 epoch mean")

plt.fill_between(inter_prec_arr[0], min_line_bound, max_line_bound, alpha=0.4)




ax.legend(line_array, names)
plt.show()

