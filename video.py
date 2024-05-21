import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl
import os
import time

print("file exists?", os.path.exists('src/test5.mp4'))

#img = cv2.imread('src/test2.mp4')
cap = cv2.VideoCapture('src/test5.mp4')
#cap = cv2.VideoCapture(0)
ticker= 0
while True:
    ticker +=1
    success, img = cap.read()
    if ticker%50 == 0:
        img_tmp = img
        #img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img
        img = cv2.bilateralFilter(img, 11, 15, 15)  # 11 15 15
        img = cv2.Canny(img, 30, 140)

        #time.sleep(5)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.bilateralFilter(img, 11, 15, 15) # 11 15 15
        # img = cv2.Canny(img, 20, 210)




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cont = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont = imutils.grab_contours(cont)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)

        pos = None
        for c in cont:
            approx = cv2.approxPolyDP(c, 10, True)
            if len(approx) == 4:
                pos = approx
                break

        print(pos)

        mask = np.zeros(gray.shape, np.uint8)
        new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
        bitwize_img = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropp = gray[x1:x2, y1:y2]

        text = easyocr.Reader(['en'])
        text = text.readtext(cropp)


        #print(text)
        if text!=[]:
            print("+mask")
            res = text[0][-2]
            print(len(res))
            if len(res) >=6:
                #if res[0].isalpha() & res[1].isdigit() & res[2].isdigit() & res[3].isdigit() & res[4].isalpha() & res[5].isalpha():
                final_image = cv2.putText(img_tmp, res, (500, 500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
                cv2.imshow('res', final_image)
                #cv2.imshow('res', img_tmp)
        else:
            cv2.imshow('res', img)

#pos = None
#for c in cont:
#    approx = cv2.approxPolyDP(c, 10, True)
#    if len(approx) == 4:
#        pos = approx
#        break

#print(pos)

#mask = np.zeros(gray.shape, np.uint8)
#new_img = cv2.drawContours(mask, [pos], 0 , 255, -1)
#bitwize_img = cv2.bitwise_and(img, img, mask=mask)


#(x, y) = np.where(mask==255)
#(x1, y1) = (np.min(x), np.min(y))
#(x2, y2 )= (np.max(x), np.max(y))
#cropp = gray[x1:x2, y1:y2]

#text = easyocr.Reader(['en'])
#text = text.readtext(cropp)

#res = text[0][-2]
#final_image = cv2.putText(img, res, (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 4)
#final_image = cv2.rectangle(img, (x1, x2),(y1, y2), (0, 255, 0), 2)

#pl.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
#pl.show()


#pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#pl.imshow(gray)
#cv2.imshow('Podoroge', img)
#cv2.waitKey()