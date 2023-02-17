import cv2
import pandas
from PIL import Image, ImageDraw
import numpy as np

csv = pandas.read_csv("/home/artem/Downloads/patterns_markup.csv")

print(csv.head())

#print(csv[csv["filename"] == "Excise_stamp_0007.jpg"])

#print(csv[csv["filename"] == "Excise_stamp_0007.jpg"]["region_shape_attributes"])


polygons = []
classes = []
img_name = "[S]ISBN_02_0064.jpg"

names = []
for name in csv["filename"]:
    names.append(name.split(".")[0])

csv["filename"] = names

print(csv["filename"] )
sample_with_images = csv[csv["filename"] == img_name.split(".")[0]]
for polygon_data in zip(sample_with_images["region_shape_attributes"], sample_with_images["region_attributes"]):
    polygon_data, polygon_class = polygon_data
    classes.append(polygon_class)
    print(polygon_class)
    print(polygon_data, polygon_class)
    try:
        _, polygon_data = polygon_data.split("all_points_x")

        polygon_data_x, polygon_data_y = polygon_data.split("all_points_y")
        print(polygon_data_x, polygon_data_y)

        polygon_data_x = polygon_data_x[3:-3]
        polygon_data_y = polygon_data_y[3:-2]

        polygon_data_x = [int(x) for x in polygon_data_x.split(",")]
        polygon_data_y = [int(x) for x in polygon_data_y.split(",")]

        print(polygon_data_x, polygon_data_y)

        polygons.append([(x, y) for x, y in zip(polygon_data_x, polygon_data_y)])
    except:
        continue

print(polygons)

base_path = "/home/artem/Image/"
ipath = base_path + img_name
img = Image.open(ipath).convert('RGB')
width, height = img.size[0], img.size[1]

mask = Image.new('L', (width, height), 0)

for polygon, class_label in zip(polygons, classes):
    ImageDraw.Draw(mask).polygon(polygon, outline=0, fill=1)


cv2.imshow("img", np.array(img) )
cv2.imshow("mask", np.array(mask) * 1.)
cv2.waitKey()