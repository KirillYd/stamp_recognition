# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from typing import List
from tensorflow.keras.models import load_model


from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.models.textrecog.preprocessors import STN
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from scipy.optimize import minimize
from statistics import geometric_mean
import dewarprectify
import math
from generate_seals import generate_dataset
from tools.CRNN import crnn_model, predict_text
from tools.text_separation import separate_text
import shutil

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='results/',
        help='Output directory of results.')
    parser.add_argument(
        '--det',
        type=str,
        default=None,
        help='Pretrained text detection algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model. '
        'If it is not specified and "det" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--rec',
        type=str,
        default=None,
        help='Pretrained text recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--kie',
        type=str,
        default=None,
        help='Pretrained key information extraction algorithm. It\'s the path'
        'to the config file or the model name defined in metafile.')
    parser.add_argument(
        '--kie-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected kie model. '
        'If it is not specified and "kie" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--save_pred',
        action='store_true',
        help='Save the inference results to out_dir.')
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Save the visualization results to out_dir.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'det', 'det_weights', 'rec', 'rec_weights', 'kie', 'kie_weights',
        'device'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def cut(points, image):

    points = np.reshape(points, (-1, 2)) ##########
    points = np.array(points, dtype=np.int32)
    #print(points)


    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255))

    result = cv2.bitwise_and(image, image, mask=mask)


    result_with_white_background = np.ones_like(image) * 255

    result_with_white_background[mask == 255] = result[mask == 255]


    x, y, w, h = cv2.boundingRect(points)
    cropped_result = result_with_white_background[y:y + h, x:x + w]

    #output_path = f'dataset/output_image/{len(points)}.jpg'
    #cv2.imwrite(output_path, result_with_white_background)
    return cropped_result

def cut2(mask_points, image):
    points = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))

    # Создаем маску
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Маска того же размера, что и изображение
    cv2.fillPoly(mask, [points], (255))  # Заполняем маску белым цветом по координатам

    # Вырезаем область текста из исходного изображения
    text_region = cv2.bitwise_and(image, image, mask=mask)

    # Создаем пустой холст
    empty_canvas = np.ones_like(image) * 255  # Пустой белый холст

    # Вставляем вырезанную область на пустой холст
    empty_canvas[mask == 255] = text_region[mask == 255]

    return empty_canvas


def get_x_y_co(xc, yc, r, points):
    x_points, y_points = points[0::2], points[1::2]
    pts_x = []
    pts_y = []
    for i in range(160, 1600):
        y = round(yc + r * math.sin(math.radians(i / 4)), 3)
        # print(math.cos(i))
        x = round(xc + r * math.cos(math.radians(i/ 4)), 3)
        if max(x_points) > x > min(x_points) and max(y_points) > y > min(y_points):
            pts_x.append(x)
            pts_y.append(y)
    return pts_x, pts_y


def get_center(points):
    xs = points[0::2]
    ys = points[1::2]

    xmean = []
    ymean = []
    rmean = []
    for i in range(0, int(len(xs)), 3):
        for j in range(1, int(len(xs)), 3):
            for k in range(2, int(len(xs)), 3):
                A = np.array([[xs[i] - xs[j], ys[i] - ys[j]], [xs[i] - xs[k], ys[i] - ys[k]]])
                B = np.array([((xs[i] ** 2 - xs[j] ** 2) + (ys[i] ** 2 - ys[j] ** 2)) / 2,
                              ((xs[i] ** 2 - xs[k] ** 2) + (ys[i] ** 2 - ys[k] ** 2)) / 2])

                try:
                    center = np.linalg.solve(A, B)

                    radius = np.sqrt((center[0] - xs[i]) ** 2 + (center[1] - ys[i]) ** 2)

                    if 10000 > center[0] > 0 and 10000 > center[1] > 0:
                        xmean.append(center[0])
                        ymean.append(center[1])
                        rmean.append(radius)

                except Exception:
                    pass

                # radius = np.sqrt((center[0] - xs[i]) ** 2 + (center[1] - ys[i]) ** 2)
                #
                # if 10000 > center[0] > 0 and 10000 > center[1] > 0:
                #     xmean.append(center[0])
                #     ymean.append(center[1])
                #     rmean.append(radius)

    return geometric_mean(xmean), geometric_mean(ymean), geometric_mean(rmean)

def get_predicted_text(img_path, model):

    predicted_text = []

    cnt = 0

    for img in os.listdir(img_path):
        image = cv2.imread(img_path + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        predictions = predict_text(model, gray)

        for char in predictions[0]:
            if char.isdigit():
                cnt += 1
        if cnt >= 4:
            cnt = 0
            image = cv2.flip(gray, -1)
            predictions = predict_text(model, image)

        plt.imshow(image, cmap='gray')
        plt.title(f'Predicted Text: {predictions[0]}')
        plt.axis('off')
        plt.show()
        predicted_text.append(predictions[0])

    return predicted_text


def main():
    num_images = 4
    output_directory = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    #generate_dataset(num_images, output_directory)

    model = crnn_model()
    model.load_weights('cp10.weights.h5')

    image_path = './dataset/images/'
    for img in os.listdir(image_path):
        init_args, call_args = parse_args()
        call_args["inputs"] = image_path + img
        ocr = MMOCRInferencer(**init_args)
        pred_dict = ocr(**call_args)
        arr = pred_dict['predictions'][0]['det_polygons']

        image = cv2.imread(image_path + img)

        rad_arr = []
        image_to_rad = dict()

        for points in arr:
            top_points = points[0:int(len(points)/2)]
            bottom_points = points[int(len(points)/2):]
            x0t,y0t,radius_t = get_center(top_points)
            x0b, y0b, radius_b = get_center(bottom_points)

            d = abs(radius_t - radius_b)

            x0, y0, radius = (x0t+x0b)/2, (y0t+y0b)/2, (radius_t+radius_b)/2


            if (image.shape[0] > x0 > 0) and (image.shape[1] > y0 > 0) :
                cut_image = cut2(points, image)


                pts_x, pts_y = get_x_y_co(x0, y0 ,radius , points) #список x, y для построения дуги
                try:
                    dewarp_image = dewarprectify.uncurve_text_tight(cut_image, pts_x, pts_y, int(x0), int(y0), round(d))
                    separate_text(dewarp_image)
                    #cv2.imwrite(f"dataset/output_image2/{img}{len(pts_x)}.jpg", dewarp_image)
                    # image_to_rad[radius] = dewarp_image

                except:
                    pass
            else:
                cut_image = cut(points, image)
                cv2.imwrite(f"dataset/output_image2/{img}_center_{len(cut_image)}.jpg", cut_image)

        img_path = "dataset/output_image2/"
        pred_words = get_predicted_text(img_path, model)
        print(pred_words)

        shutil.rmtree("dataset/output_image2")
        os.mkdir("dataset/output_image2")


if __name__ == '__main__':
    main()



#python tools/infer.py dataset/test.jpg --det drrg