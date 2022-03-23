import os
import glob
import argparse

import cv2
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('save_path', type=str)

    args = parser.parse_args()
    results_path = args.path
    base_save_path = args.save_path

    if os.path.exists(base_save_path) is False:
        os.makedirs(base_save_path)

    results_items_paths = glob.glob(os.path.join(results_path, "*"))

    for res_item_path in results_items_paths:
        json_path = os.path.join(res_item_path, 'results.json')
        img_path = os.path.join(res_item_path, 'image.jpeg')

        image = cv2.imread(img_path)

        with open(json_path) as f:
            results_dict = json.load(f)

        for key, value in results_dict.items():  # .items() - doens't work(strange)
            left, right, top, bottom = value["box"]
            model_name = value["model_name"]
            crop_img = image[top:bottom, left:right]

            img_save_path = os.path.join(base_save_path, key + "_" + model_name + ".jpeg")
            cv2.imwrite(img_save_path, crop_img)