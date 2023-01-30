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

def gather_data(path, prec_arr, rec_arr):
    global min_rec_infer
    data_plot2 = np.load(path)
    prec_arr.append(data_plot2[:, 3] / (data_plot2[:, 3] + data_plot2[:, 2]))
    rec_arr.append(data_plot2[:, 3] / (data_plot2[:, 3] + data_plot2[:, 4]))

    print(data_plot2[:, 3])
    print(data_plot2[:, 2])
    print(data_plot2[:, 4])
    print(data_plot2[:, 5])
    exit(0)
    if min_rec_infer < rec_arr[-1][-1]:
        min_rec_infer = rec_arr[-1][-1]

    #print(np.around(data_plot2[:, 4], decimals=3) + np.around(data_plot2[:, 2], decimals=3))
    return np.around(data_plot2[:, 4], decimals=3)


base_path = "/home/artem/PycharmProjects/backboned-unet-new/plots/"
plots_num = 1
def gather_th(path):
    prec, th = [], []
    for i in range(plots_num):
        data_plot2 = np.load(base_path + path + f"/{str(i)}.npy")

        prec.append(np.around(data_plot2[:, 0], decimals=3))
        th.append(np.around(data_plot2[:, 5], decimals=3))
    return np.array(prec), np.array(th)


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


min_prec, max_prec = 0.85 - 0.025, 0.85 + 0.025

def plot_data(name, display_name):
    prec_arr, rec_arr, fn = [], [], []

    len_fn = 0
    for i in range(plots_num):
        res_fn = gather_data(base_path + name + f"/{str(i)}.npy", prec_arr, rec_arr)
        fn.append( res_fn )
        #len_fn += res.shape[0]

    # for i in range(plots_num):
    #     gather_data(base_path + name + f"/{str(i)}.npy", prec_arr, rec_arr)
    #     #len_fn += res.shape[0]

    prec_arr = np.array(prec_arr)
    rec_arr = np.array(rec_arr)

    inter_prec_arr = []
    inter_rec_arr = []
    inter_fn = []

    for line_x, line_y, line_fn in zip(rec_arr, prec_arr, fn):
        inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
        inter_line_x, inter_line_fn = get_interpolation_line(line_x, line_fn)

        inter_rec_arr.append(inter_line_x)
        inter_prec_arr.append(inter_line_y)
        inter_fn.append(inter_line_fn)

    inter_prec_arr = np.array(inter_prec_arr)
    inter_rec_arr = np.array(inter_rec_arr)
    inter_fn = np.array(inter_fn)

    min_line_bound, max_line_bound = get_bounds(inter_prec_arr)

    prec_mean = np.mean(inter_prec_arr, axis=0)
    rec_mean = np.mean(inter_rec_arr, axis=0)
    fn_mean = np.mean(inter_fn, axis=0)

    line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
    line_array.append(line)

    names.append(display_name)


    prec_mean_70_80 = prec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
    rec_mean_70_80 = rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
    fn_mean_70_80 = fn_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]

    dist = np.mean(rec_mean[(max_line_bound > min_prec) & (max_line_bound < max_prec)]) - np.mean(rec_mean[(min_line_bound > min_prec) & (min_line_bound < max_prec)])

    max_distance.append(dist)

    prec_mean_70_80_val = np.mean(prec_mean_70_80)
    rec_mean_70_80_val = np.mean(rec_mean_70_80)
    fn_mean_70_80_val = np.mean(fn_mean_70_80)

    min_fn_line_bound, max_fn_line_bound = get_bounds(inter_fn)
    min_fn_mean_70_80 = np.mean(min_fn_line_bound[(prec_mean > min_prec) & (prec_mean < max_prec)])
    max_fn_mean_70_80 = np.mean(max_fn_line_bound[(prec_mean > min_prec) & (prec_mean < max_prec)])

    print("fn", min_fn_mean_70_80, fn_mean_70_80_val, max_fn_mean_70_80)
    max_prec_array = np.max(inter_prec_arr, axis=0)
    print(display_name+" average max_rec ", np.mean(rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]) )
    print(display_name + " max max_rec ", np.mean(rec_mean[(max_line_bound > min_prec) & (max_line_bound < max_prec)]))
    print(display_name + " min max_rec ", np.mean(rec_mean[(min_line_bound > min_prec) & (min_line_bound < max_prec)]))

    print(display_name + "f1:",  2 * prec_mean_70_80_val * rec_mean_70_80_val / (prec_mean_70_80_val + rec_mean_70_80_val) )
    print(display_name + "f2:",  (1 + 2**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((2**2) * prec_mean_70_80_val + rec_mean_70_80_val) )
    print(display_name + "f5:",  (1 + 5**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((5**2) * prec_mean_70_80_val + rec_mean_70_80_val) )

    plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)

# name = "f1+100+syntetic"
# display_name = "f1 версия 1"
# plot_data(name, display_name)
#
# name = "Original+200+syntetic"
# display_name = "Original syntetic"
# plot_data(name, display_name)
#
# name = "resnetCenterNetUnet+100+syntetic"
# display_name = "resnetCenterNetUnet syntetic"
# plot_data(name, display_name)

#
# name = "resnetCenterNetUnet+100+syntetic"
# display_name = "resn"
# plot_data(name, display_name)


name = "f1+100+syntetic"
display_name = "f1+100+syntetic"
plot_data(name, display_name)




#
# name = "Original+200+syntetic"
# display_name = "Original+200+syntetic"
# plot_data(name, display_name)


#prec, th = gather_th(name)
#print(np.max(th[(prec > min_prec) & (prec < max_prec)]))

# name = "original+100"
# display_name = "Оригинал"
# plot_data(name, display_name)
#
# name = "original+200"
# display_name = "Оригинал x2"
# plot_data(name, display_name)


#
#
#
# name = "original+100"
# display_name = "Оригинал"
# plot_data(name, display_name)
# prec, th = gather_th(name)
# print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
#
# name = "f1+100+pretrainCenterNet"
# display_name = "f1 версия 2 pretrainCenterNet (Многоцелевая)"
# plot_data(name, display_name)
# prec, th = gather_th(name)
# print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
# #
# # name = "f1+100+adapt+pretrainCenterNet"
# # display_name = "f1 версия 2 (Многоцелевая, адаптивная)"
# # plot_data(name, display_name)
# # prec, th = gather_th(name)
# # print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
# name = "f1+100"
# display_name = "f1 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
# #
# name = "f2+100"
# display_name = "f2 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
#
#
# # name = "f1+100+adapt+pretrainCenterNet"
# # display_name = "f1 версия 2 f1+100+adapt+pretrainCenterNet (Многоцелевая)"
# # plot_data(name, display_name)
# #
# # name = "f1+100+pretrainCenterNet"
# # display_name = "f1 версия 2  f1+100+pretrainCenterNet(Многоцелевая)"
# # plot_data(name, display_name)
#
# #prec, th = gather_th(name)
# #print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
# # name = "all/f2+resNetInitFalse+v2+100"
# # display_name = "f2 версия 2 (Многоцелевая)"
# # plot_data(name, display_name)
#
# #
# name = "f5+100"
# display_name = "f5 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
#
# name = "f5+v2+100+pretrainCenterNet"
# display_name = "f5 версия 2 pretrainCenterNet (Многоцелевая)"
# plot_data(name, display_name)
#
#
# #
#
# name = "all/f2+100+pretrainCenterNet"
# display_name = "f2 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
#prec, th = gather_th(name)
#print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
#
# name = "f5+v2+100+pretrainCenterNet"
# display_name = "f5 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
# prec, th = gather_th(name)
# print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
# #
# name = "f5+v2+100+pretrainCenterNet"
# display_name = "f5 версия 2 (Многоцелевая)"
# plot_data(name, display_name)
# #
#
# #
# # name = "all/f2+resnetInitFalse+v2+30"
# # display_name = "f2 кол-во эпох 30"
# # plot_data(name, display_name)
# # #
# # name = "all/f2+resnetInitFalse+v2"
# # display_name = "f2 кол-во эпох 10"
# # plot_data(name, display_name)
#
#
# #
# #
# # name = "f5+resnetInitFalse+v2+100"
# # plot_data(name)
#
# # name = "testNewFeatures1+100"
# # display_name = "testNewFeatures1"
# # plot_data(name, display_name)
# # prec, th = gather_th(name)
# # print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
# name = "resnetCenterNetUnet+100"
# display_name = "Оригинал + учет размеров штрихкода"
# plot_data(name, display_name)
# prec, th = gather_th(name)
# print(np.max(th[(prec > min_prec) & (prec < max_prec)]))
#
#
# name = "testNewFeatures1+100"
# display_name = "Оригинал + учет размеров и признаков штрихкода"
# plot_data(name, display_name)

#
# name = "AdaptCenterNetUnet+100"
# display_name = "Оригинал + адаптация"
# plot_data(name, display_name)
# prec, th = gather_th(name)
# print(np.max(th[(prec > min_prec) & (prec < max_prec)]))


# prec_arr, rec_arr = [], []
# name = "resnetCenterNetUnet+100"
# fn = 0
# len_fn = 0
# for i in range(10):
#     res = gather_data("/home/artem/PycharmProjects/backboned-unet-new/plots/"
#                 + name + f"/{str(i)}.npy", prec_arr, rec_arr)[0]
#     fn += np.sum(res)
#     #len_fn += res.shape[0]
#
# fn_array.append(fn / 10)

# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_rec_arr.append(inter_line_x)
#     inter_prec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_prec_arr)
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# max_prec_array = np.max(inter_prec_arr, axis=0)
# print(name+"max rec ", np.mean(rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]) )
#
#
# line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
# line_array.append(line)
#
# names.append(name)
#
# prec_mean_70_80 = prec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
# rec_mean_70_80 = rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
#
# dist = np.mean(rec_mean[(max_line_bound > min_prec) & (max_line_bound < max_prec)]) -np.mean(rec_mean[(min_line_bound > min_prec) & (min_line_bound < max_prec)])
#
# max_distance.append(dist)
#
#
# prec_mean_70_80_val = np.mean(prec_mean_70_80)
# rec_mean_70_80_val = np.mean(rec_mean_70_80)
#
# print(1/rec_mean_70_80_val - 1)
#
#
# f1 = 2 * prec_mean_70_80_val * rec_mean_70_80_val / (prec_mean_70_80_val + rec_mean_70_80_val)
# f2 = (1 + 2**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 2**2) * prec_mean_70_80_val + rec_mean_70_80_val)
# f5 = (1 + 5**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 5**2) * prec_mean_70_80_val + rec_mean_70_80_val)
#
# print(name + "f1:", f1)
# print(name + "f2:", f2)
# print(name + "f5:", f5)
#
# plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)






#
# prec_arr, rec_arr = [], []
# name = "testNewFeatures+100"
# fn = 0
# len_fn = 0
# for i in range(10):
#     res = gather_data("/home/artem/PycharmProjects/backboned-unet-new/plots/"
#                 + name + f"/{str(i)}.npy", prec_arr, rec_arr)[0]
#     fn += np.sum(res)
#     #len_fn += res.shape[0]
#
# fn_array.append(fn / 10)

# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_rec_arr.append(inter_line_x)
#     inter_prec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_prec_arr)
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# max_prec_array = np.max(inter_prec_arr, axis=0)
# print(name+"max rec ", np.mean(rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]) )
#
# line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
# line_array.append(line)
#
# names.append(name)
#
# prec_mean_70_80 = prec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
# rec_mean_70_80 = rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
#
# dist = np.mean(rec_mean[(max_line_bound > min_prec) & (max_line_bound < max_prec)]) -np.mean(rec_mean[(min_line_bound > min_prec) & (min_line_bound < max_prec)])
#
# max_distance.append(dist)
#
#
# prec_mean_70_80_val = np.mean(prec_mean_70_80)
# rec_mean_70_80_val = np.mean(rec_mean_70_80)
#
# print(1/rec_mean_70_80_val - 1)
#
#
# f1 = 2 * prec_mean_70_80_val * rec_mean_70_80_val / (prec_mean_70_80_val + rec_mean_70_80_val)
# f2 = (1 + 2**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 2**2) * prec_mean_70_80_val + rec_mean_70_80_val)
# f5 = (1 + 5**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 5**2) * prec_mean_70_80_val + rec_mean_70_80_val)
#
# print(name + "f1:", f1)
# print(name + "f2:", f2)
# print(name + "f5:", f5)
#
# plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)
#
#






#
#
# prec_arr, rec_arr = [], []
# name = "AdaptCenterNetUnet+100"
# fn = 0
# len_fn = 0
# for i in range(10):
#     res = gather_data("/home/artem/PycharmProjects/backboned-unet-new/plots/"
#                 + name + f"/{str(i)}.npy", prec_arr, rec_arr)[0]
#     fn += np.sum(res)
#     #len_fn += res.shape[0]
#
# fn_array.append(fn / 10)

# prec_arr = np.array(prec_arr)
# rec_arr = np.array(rec_arr)
#
# inter_prec_arr = []
# inter_rec_arr = []
#
# for line_x, line_y in zip(rec_arr, prec_arr):
#     inter_line_x, inter_line_y = get_interpolation_line(line_x, line_y)
#
#     inter_rec_arr.append(inter_line_x)
#     inter_prec_arr.append(inter_line_y)
#
# inter_prec_arr = np.array(inter_prec_arr)
# inter_rec_arr = np.array(inter_rec_arr)
#
# min_line_bound, max_line_bound = get_bounds(inter_prec_arr)
#
# prec_mean = np.mean(inter_prec_arr, axis=0)
# rec_mean = np.mean(inter_rec_arr, axis=0)
#
# line, = ax.plot(rec_mean, prec_mean, **line_kwargs)
# line_array.append(line)
#
# names.append(name)
#
# prec_mean_70_80 = prec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
# rec_mean_70_80 = rec_mean[(prec_mean > min_prec) & (prec_mean < max_prec)]
#
#
# dist = np.mean(rec_mean[(max_line_bound > min_prec) & (max_line_bound < max_prec)]) -np.mean(rec_mean[(min_line_bound > min_prec) & (min_line_bound < max_prec)])
#
# max_distance.append(dist)
#
#
# prec_mean_70_80_val = np.mean(prec_mean_70_80)
# rec_mean_70_80_val = np.mean(rec_mean_70_80)
#
# print(1/rec_mean_70_80_val - 1)
#
#
# f1 = 2 * prec_mean_70_80_val * rec_mean_70_80_val / (prec_mean_70_80_val + rec_mean_70_80_val)
# f2 = (1 + 2**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 2**2) * prec_mean_70_80_val + rec_mean_70_80_val)
# f5 = (1 + 5**2) * prec_mean_70_80_val * rec_mean_70_80_val / ((1 + 5**2) * prec_mean_70_80_val + rec_mean_70_80_val)
#
# print(name + "f1:", f1)
# print(name + "f2:", f2)
# print(name + "f5:", f5)
#
# plt.fill_between(inter_rec_arr[0], min_line_bound, max_line_bound, alpha=0.4)
#







plt.fill_between([0, 1], min_prec, max_prec, alpha=0.4)

#plt.fill_between([0, min_rec_infer], 0, 1, alpha=0.4)

mean_fn = []

#print(fn_array[3] / np.array(fn_array) )

print(max_distance)
#print(max_distance[3] / max_distance )
ax.legend(line_array, names)
plt.show()

