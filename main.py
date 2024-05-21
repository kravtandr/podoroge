import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as pl
import os
print("file exists?", os.path.exists('src/test6.jpg'))

img = cv2.imread('src/test7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



img_filter = cv2.bilateralFilter(gray, 11, 15, 15) # 11 15 15

img = cv2.GaussianBlur(img, (3,3), 0)

edges = cv2.Canny(img_filter, 30, 140)

cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key=cv2.contourArea, reverse=True)

pos = None
for c in cont:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        print("FIND!!!!!!")
        pos = approx
        break

#print(pos)

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0 , 255, -1)
bitwize_img = cv2.bitwise_and(img, img, mask=mask)



(x, y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2 )= (np.max(x), np.max(y))
cropp = gray[x1:x2, y1:y2]

text = easyocr.Reader(['en'])
text = text.readtext(cropp)
print(text)


res = text[0][-2]
final_image = cv2.putText(img, res, (500, 500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
final_image = cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), 2)




cv2.imshow('Podoroge', final_image)
cv2.waitKey(5000)
cv2.imshow('Podoroge', cropp)
cv2.waitKey()



#pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#pl.imshow(gray)
#cv2.imshow('Podoroge', img)
#cv2.waitKey()