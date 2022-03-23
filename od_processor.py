import time

import os
import glob
import json
from shutil import rmtree

import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# import tensorflow_hub as hub  # will be needed for futher experiments

from od_utils import resize_with_padding, ALL_MODELS, get_car_type_model, get_average_color


class ODProcessor:
    """
    Class for cars data processing
        classes_txt_file - path to txt file which is contains names of cars models
        od_w_path - path to object detection (od) model's weights
        car_type_w_path - path to model which is clasify type (model) of car
        base_save_path - path for storing boxes, pass None if you don't want to

        USEFUL URLS:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb#scrollTo=2O7rV8g9s8Bz
    """

    def __init__(self,
                 classes_txt_file= "object_detection/trunc_car_models.txt",
                 od_w_path="object_detection/models/SSD MobileNet v2 320x320",
                 car_type_w_path="object_detection/models/CarTypeMobileNetV2/MobileNetV2.ckpt",
                 base_save_path="object_detection/results"):

        self.base_save_path = base_save_path

        self.model_classes = []
        with open(classes_txt_file) as f:
            for line in f.readlines():
                self.model_classes.append(line)

        # 3 car 4 motocycle 6 bus 8 track
        self.car_classes = [3, 4, 6, 8]

        # will be needed for futher experiments
        # model = hub.load(ALL_MODELS["CenterNet HourGlass104 512x512"])
        # tf.saved_model.save(model, "object_detection/models/CenterNetHourGlass104512x512")

        self.od_model = tf.saved_model.load(od_w_path)
        self.car_type_model = get_car_type_model(w_path=car_type_w_path,
                                                 n_classes=len(self.model_classes))

    # (!) add async if you need coroutine object (!)
    def run_od_inference(self, image_np):
        start_time = time.time()
        results = self.od_model(image_np)
        print("time", time.time() - start_time)

        # different object detection models have additional results
        # all of them are explained in the documentation
        result = {key: value.numpy() for key, value in results.items()}

        img_w = image_np.shape[2]
        img_h = image_np.shape[1]

        detection_boxes = result["detection_boxes"][0, :, :]  # remove batch dim TEMP (can be fixed in future)
        detection_scores = result["detection_scores"][0, :]  # remove batch dim TEMP
        detection_classes = result['detection_classes'][0]  # remove batch dim TEMP

        cropped_boxes = []

        for index, box in enumerate(detection_boxes):

            # filter low prob boxes  and unneeded classes
            if detection_scores[index] < 0.4 or detection_classes[index] not in self.car_classes:
                continue

            y1, x1, y2, x2 = box

            x1 *= img_w
            x2 *= img_w

            y1 *= img_h
            y2 *= img_h

            left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)  # rename for convievence

            # box_w = right - left
            # box_h = bottom - top

            # if (box_w * box_h) / (img_w * img_h) < 0.1:  # filter boxes by relative area to image
            #     continue

            cropped_boxes.append([left, right, top, bottom])

        return cropped_boxes

    def process(self, image):
        buff_result_dict = {}
        ts = time.time()

        print("process start", ts)

        cropped_boxes = self.run_od_inference(np.expand_dims(image, axis=0))

        result_dict = {}
        for index, crop_box in enumerate(cropped_boxes):
            left, right, top, bottom = crop_box

            orig_crop_img = image[top:bottom, left:right]
            crop_img = Image.fromarray(np.uint8(orig_crop_img))  # .convert('RGB')
            crop_img = resize_with_padding(crop_img, (224, 224))
            crop_img = preprocess_input(np.asarray(crop_img))

            preds = self.car_type_model(np.expand_dims(crop_img, axis=0), training=False)
            pred_class_name = self.model_classes[np.argmax(preds)]

            crop_box_key = "{}_{}".format(ts, index)

            result_dict[crop_box_key] = pred_class_name

            item_dict = {}
            item_dict["model_name"] = pred_class_name
            item_dict["box"] = [left, right, top, bottom]
            item_dict["colors"] = get_average_color(crop_img)

            buff_result_dict[crop_box_key] = item_dict

        if self.base_save_path is not None:
            self.store_results(ts, image, buff_result_dict)

        return buff_result_dict

    def store_results(self, timestamp, image, results_dict):
        res_item_path = os.path.join(self.base_save_path, str(timestamp))
        os.makedirs(res_item_path)

        json_path = os.path.join(res_item_path, 'results.json')
        img_path = os.path.join(res_item_path, 'image.jpeg')

        cv2.imwrite(img_path, image)

        with open(json_path, 'w') as fp:
            json.dump(results_dict, fp)

    def visualize_results(self):
        results_paths = glob.glob(os.path.join(self.base_save_path, "*"))

        for res_item_path in results_paths:
            json_path = os.path.join(res_item_path, 'results.json')
            img_path = os.path.join(res_item_path, 'image.jpeg')

            img = cv2.imread(img_path)

            with open(json_path) as f:
                results_dict = json.load(f)

            for _, value in results_dict.items():  #.items() - doens't work(strange)
                left, right, top, bottom = value["box"]
                model_name = value["model_name"]

                img = cv2.rectangle(img, (left, top), (right, bottom), (36, 255, 12), 2)
                cv2.putText(img, model_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

            cv2.imshow("test1", img)
            cv2.waitKey(0)

def clear_results(base_save_path):
    folders = glob.glob(os.path.join(base_save_path, "*"))
    for f in folders:
        rmtree(f)

def test_on_test_imgs():
    base_save_path = "object_detection/results"
    processor = ODProcessor(base_save_path=base_save_path)

    #  Temp
    clear_results(base_save_path=base_save_path)
    #

    test_img_paths = glob.glob(os.path.join("object_detection/od_test_data", "*"))
    for img_path in test_img_paths:
        img = cv2.imread(img_path)

        _ = processor.process(img)

    processor.visualize_results()

if __name__ == "__main__":
    test_on_test_imgs()
