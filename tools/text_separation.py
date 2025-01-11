import cv2
import numpy as np
from numpy import array

import matplotlib.pyplot as plt
# Загрузка изображения


def cut_words(hist):
    #print(hist)
    hist = np.array(hist).tolist()
    #print(sum(hist)/255)
    null_cnt = 0
    black_pix = False
    cut_x = [0]
    a = 0
    for i in range(len(hist)):
        if (null_cnt >= 15) and (hist[i]/255 > 15):
            cut_x.append(i-null_cnt//2)
            null_cnt = 0

        if black_pix and ((hist[i]/255)<= 7):
            null_cnt +=1
            a+=1

        if (hist[i]/255 > 15):
            null_cnt = 0
            #print("iiiiiiiiiiii", i)
            black_pix = True

    # for i in range(156, 171):
    #     print(hist[i])
    # print(black_pix)
    # print("a", a)


    cut_x.append(len(hist))
    return cut_x


def save_cut_image(img, cut_x):
    # print("2"*10)
    for i in range(len(cut_x) - 1):
        img2 = img[0:img.shape[1],cut_x[i]:cut_x[i+1]]
        #print("img2", img2)
        cut = [str(i) for i in cut_x]
        text = "".join(cut)

        cv2.imwrite(f"dataset/output_image2/{text}_{i}.jpg", img2)
        #cv2.imshow("123",img2)
        #cv2.waitKey(0)


def find_hist(binary):
    hist = [0] * len(binary[0])
    for x in binary:
        hist = np.add(hist, np.array(x).tolist())
    return hist


def separate_text(image):
    #print('1'*10)
    #print("image\n", image)
    #image = cv2.imread(image_path)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print("gray\n",gray)

    # Бинаризация изображения
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #print("binary\n", binary)


    hist = find_hist(binary)

    black_pixels = sum(hist) / 255
    pixels_cnt = image.shape[0] * image.shape[1]
    #print("aaaaaaaaaaaaaaaaa", black_pixels/pixels_cnt)

    if black_pixels/pixels_cnt > 0.40:
        img_neg = cv2.bitwise_not(image)
        #_, binary = cv2.threshold(img_neg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print(img_neg)
        rev_hist = find_hist(img_neg)
        cut_x = cut_words(rev_hist)
        #print("cut x")
        #print(cut_x)
        save_cut_image(img_neg, cut_x)
    else:
        _, img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x_hist = find_hist(img)
        cut_x = cut_words(hist)
       # print("cut x")
       # print(cut_x)
        # kernel = np.ones((3, 3), np.uint8)
        # thresh = cv2.erode(img, kernel, iterations=1)
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        save_cut_image(img, cut_x)






# x = [i+1 for i in range(len(hist))]
#     print(x)
 # plt.plot(x, hist)
    # plt.show()
    # #plt.imshow()
    #
    # # Найти контуры
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilated = cv2.dilate(binary, kernel, iterations=1)
    # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Хранилище для изображений слов
    # word_images = []
    #
    # for contour in contours:
    #     # Получаем координаты ограничивающего прямоугольника
    #     x, y, w, h = cv2.boundingRect(contour)
    #
    #     # Фильтруем слишком маленькие или большие области
    #     if w > 10 and h > 10:
    #         # Вырезаем слово
    #         word_image = image[y:y + h, x:x + w]
    #         word_images.append(word_image)
    #
    #         # Отображаем границы на изображении (для визуализации)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # Сохраняем обработанное изображение (для проверки)
    # cv2.imwrite("words_detected.jpg", image)
    #
    # # Теперь можно отправить каждую word_image в CRNN для распознавания
    # #for idx, word_image in enumerate(word_images):
    # #    cv2.imwrite(f"word_{idx}.jpg", word_image)