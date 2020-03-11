import glob
import xml.etree.ElementTree as ET
import os
import numpy as np

from code import kmeans

kmeans_ = kmeans.kmeans
avg_iou = kmeans.avg_iou

ANNOTATIONS_PATH = "./data/images&xml/xml/"
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    for file in os.listdir(path):
        for xml_file in glob.glob("{}/*xml".format(path + file)):

            tree = ET.parse(xml_file)

            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):
                try:
                    xmin = int(obj.findtext("bndbox/xmin")) / width
                    ymin = int(obj.findtext("bndbox/ymin")) / height
                    xmax = int(obj.findtext("bndbox/xmax")) / width
                    ymax = int(obj.findtext("bndbox/ymax")) / height
                except:
                    continue

                dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)


# data = load_dataset(ANNOTATIONS_PATH)
# print('kmeans...')
# out = kmeans_(data, k=CLUSTERS)
# print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
# print("Boxes:\n {}".format(out))
#
# ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
# print("Ratios:\n {}".format(sorted(ratios)))
'''
Accuracy: 68.23%
Boxes:
 [[0.83666667 0.81666667]
 [0.22375    0.43333333]
 [0.145      0.145     ]
 [0.29375    0.21083333]
 [0.0325     0.03083333]
 [0.432      0.66875   ]
 [0.76875    0.48583333]
 [0.07875    0.06875   ]
 [0.46444444 0.35      ]]
Ratios:
 [0.52, 0.65, 1.0, 1.02, 1.05, 1.15, 1.33, 1.39, 1.58]
 '''

box = [[0.83666667, 0.81666667],
       [0.22375, 0.43333333],
       [0.145, 0.145],
       [0.29375, 0.21083333],
       [0.0325, 0.03083333],
       [0.432, 0.66875],
       [0.76875, 0.48583333],
       [0.07875, 0.06875],
       [0.46444444, 0.35]]
print(np.array(box) * 416)
