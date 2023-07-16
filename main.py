# This is a sample Python script.
import random

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    step = 0
    while capture.isOpened():
        _, image = capture.read()
        image = cv2.flip(image, 1)
        image = np.array(image)
        # step += 1
        # F = (step % 32) + 1
        # original_size = (image.shape[1], image.shape[0])
        # image = cv2.resize(image, (image.shape[1] // F, image.shape[0] // F))
        # image = cv2.resize(image, original_size, 0, 0, 0, cv2.INTER_CUBIC)
        original_image: cv2.UMat = image

        detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # cv2.circle(original_image, (360, 100), 160, (28 + step % 10 * 10, 120 + step % 10 * 10, 255 - step % 10 * 10), 17, cv2.LINE_AA)
        # blur_kernel = cv2.getGaussianKernel(7, 1.0)
        # image = cv2.blur(image, (15, 15))
        image = cv2.Canny(image, 32, 32)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype(np.uint8)
        print("image:", image.shape, image.dtype)
        print("original_image:", original_image.shape, original_image.dtype)
        image = cv2.vconcat((original_image, image))
        cv2.imshow("Camera", image)
        c = cv2.waitKey(1)
        if c == 27:
            break



