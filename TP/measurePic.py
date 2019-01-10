from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to Image")
ap.add_argument("-w", "--width", type=float, required=True, help=" Width")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

imghsv = cv2.GaussianBlur(imghsv, (7, 7), 0)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

_, imghsv = cv2.threshold(imghsv,100,150, cv2.THRESH_TOZERO_INV)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

imghsv = cv2.Canny(imghsv, 100, 150)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

imghsv = cv2.dilate(imghsv, None, iterations=10)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

imghsv = cv2.erode(imghsv, None, iterations=8)
cv2.imshow("Image", imghsv)
cv2.waitKey(0)

cnts = cv2.findContours(imghsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[1]

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
    print(cv2.contourArea(c))
    if cv2.contourArea(c) < 500:
        continue

    orig = image.copy() 
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    cv2.putText(orig, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    cv2.imshow("Image", orig)
    cv2.waitKey(0)
