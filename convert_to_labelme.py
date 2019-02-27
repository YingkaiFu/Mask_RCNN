import argparse
import json
from PIL import Image
import os
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from skimage.measure import find_contours
from shapely.geometry import Polygon

ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



def main():
    parser = argparse.ArgumentParser(description="Convert images to json with Mask-RCNN")
    parser.add_argument('--thre', default=0.8, type=int, help='Detection threshold')
    parser.add_argument('--src', default='src', help='Image dir')
    parser.add_argument('--out', default='out', help='Output dir')
    args = parser.parse_args()
    converter = Converter(args)
    converter.run()


class Converter(object):
    def __init__(self, config):
        self.src = config.src
        self.thre = config.thre
        self.out = config.out
        self.file_list = None
        self.version = "3.8.1"
        self.flags = {}
        self.shape = []
        self.lineColor = [0, 255, 0, 128]
        self.fillColor = [255, 0, 0, 128]
        self.imageData = None
        self.imagePath = None
        self.imageHeight = None
        self.imageWidth = None
        assert os.path.exists(self.src), '输入文件夹不存在'
        assert (os.path.exists(self.out) and not os.listdir(self.out)) or (not os.path.exists(self.out)), '输出文件夹不为空'
        if not os.path.exists(self.out):
            os.makedirs(self.out)

    def getfile_index(self):
        for root, dirs, files in os.walk(self.src):
            self.file_list = files

    def run(self):
        # 获取一个文件索引
        self.getfile_index()
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        # 对于每个文件，获取其对应的mask列表，识别出的结果列表
        for num, image_name in enumerate(self.file_list):
            image_dir = self.src + '/' + image_name
            self.init_variables()
            self.imagePath = image_name
            self.imageHeight, self.imageWidth, channel = self.add_image(image_name,num)
            # Todo: 对于新识别出的种类，将其加入到categories中，此处有个循环
            image = skimage.io.imread(image_dir)
            results = model.detect([image], verbose=1)
            r = results[0]
            self.get_shape(r)

            # r = result[0]
            # label = r['class_ids']
            # point = self.get_points(r['masks'])
            #
            # self.categories.append(label)
            # self.annotations.append(self.annotation(point, label, num))
            coco = self.data2coco()
            output_name = image_name[0:-4] + ".json"
            json.dump(coco, open(self.out+"/"+output_name, 'w'), indent=4)

    def detection(self, image):
        return True, False

    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        poly = Polygon(points)
        area_ = round(poly.area, 6)
        annotation['area'] = area_
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float, self.getbbox(points)))

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def data2coco(self):
        data_coco = {}
        data_coco['version'] = self.version
        data_coco['flags'] = self.flags
        data_coco['shapes'] = self.shape
        data_coco['lineColor'] = self.lineColor
        data_coco['fillColor'] = self.fillColor
        data_coco['imagePath'] = self.imagePath
        data_coco['imageData'] = self.imageData
        data_coco['imageHeight'] = self.imageHeight
        data_coco['imageWidth'] = self.imageWidth
        # data_coco['categories'] = self.categories
        # data_coco['annotations'] = self.annotations
        return data_coco

    def get_points(self, image):
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def add_image(self, file_name, num):
        image = {}
        image_dir = self.src + '/' + file_name
        img = skimage.io.imread(image_dir)
        return img.shape

    def init_variables(self):
        self.imageData = None
        self.imagePath = None
        self.imageHeight = None
        self.imageWidth = None

    def get_shape(self, r):
        N = r['rois'].shape[0]
        shapes = None
        self.shape = []
        points = None
        for i in range(N):
            shapes = {}
            points = []
            class_id = r['class_ids'][i]
            label = class_names[class_id]
            mask = r['masks'][:, :, i]
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                num, col = verts.shape
                n = num-num%15
                out = np.zeros([15,2])
                temp = np.linspace(0,n,15,endpoint=False)
                for i in range(15):
                    out[i] = verts[int(temp[i])]
                points.extend(out.tolist())
            shapes['label'] = label
            shapes['line_color'] = None
            shapes['fill_color'] = None
            shapes['points'] = points
            shapes['shape_type'] = 'polygon'
            self.shape.append(shapes)


if __name__ == '__main__':
    main()
