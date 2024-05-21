import cv2
import pytesseract
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl
import os
import time

def ConvertImg(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    final_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.medianBlur(final_image, 3)  # kernel size 3
    return final_image

def DrawTable(img_tmp):
    cv2.rectangle(img_tmp, (95, 610), (450, 550), (0, 215, 0), 2)
    cv2.rectangle(img_tmp, (95, 610), (340, 550), (0, 215, 0), 2)

def isPlate(text):
    if len(text) >= 8:
        if text[0] == 3:
            text[0] = 'E'
        if text[4] == 3:
            text[4] = 'E'
        if text[5] == 3:
            text[5] = 'E'
        if text[0] == 8:
            text[0] = 'B'
        if text[4] == 8:
            text[4] = 'B'
        if text[5] == 8:
            text[5] = 'B'
        if text[0].isalpha() & text[1].isdigit() & text[2].isdigit() & text[3].isdigit() & text[4].isalpha() & text[5].isalpha() & text[6].isdigit() & text[7].isdigit():

            return (True, text[:6], text[6:9])
    return (False, text)
def main():
    print("file exists?", os.path.exists('src/test5.mp4'))
    cap = cv2.VideoCapture('src/test5.mp4')
    # cap = cv2.VideoCapture(0)
    ticker = 0
    while True:
        ticker += 1
        success, img = cap.read()
        if ticker % 80 == 0:
            img_tmp = img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = img

            plate = cv2.CascadeClassifier('plate.xml')
            results = plate.detectMultiScale(img, scaleFactor=1.5, minNeighbors=2)

            cropp = img_tmp
            for (x, y, w, h) in results:
                cv2.rectangle(img_tmp, (x, y), (x + w, y + h), (0, 255, 0), 3)

                cropp = img[y:y + h, x:x + w]

                plate_opt = ConvertImg(cropp, 200)
                text = pytesseract.image_to_string(plate_opt,
                                                   config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=BCDFGHJKLMNEPQRSTVWXYZ0123456789')
                print(text)
                if text != []:
                    print("+mask")
                    if isPlate(text)[0] == True:
                        plate_width, plate_height = cropp.shape
                        img_tmp[500:500+plate_width, 95:95+plate_height] = img_tmp[y:y + h, x:x + w]
                        cv2.putText(img_tmp, isPlate(text)[1], (100, 600), cv2.FONT_HERSHEY_PLAIN, 3,
                                                  (0, 255, 0), 4)
                        cv2.putText(img_tmp, isPlate(text)[2], (345, 600), cv2.FONT_HERSHEY_PLAIN, 3,
                                                  (0, 255, 0), 4)
                    DrawTable(img_tmp)

            cv2.imshow('res', img_tmp)
            # cv2.imshow('res', cropp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    # execute only if run as a script
    main()