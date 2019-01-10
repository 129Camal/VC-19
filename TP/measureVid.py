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
ap.add_argument("-w", "--width", type=float, required=True, help=" Width")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)

while(True):
    ret, image = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    imghsv = cv2.GaussianBlur(imghsv, (7, 7), 0)

    _, imghsv = cv2.threshold(imghsv,100, 150, cv2.THRESH_TOZERO_INV)

    imghsv = cv2.Canny(imghsv, 100, 150)

    imghsv = cv2.dilate(imghsv, None, iterations=1)

    #imghsv = cv2.erode(imghsv, None, iterations=1)

    cv2.imshow("input", imghsv)

    cnts = cv2.findContours(imghsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[1]

    pixelsPerMetric = None

    for c in cnts:
        print(cv2.contourArea(c))
        if cv2.contourArea(c) < 5000:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

    
        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / args["width"]

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        cv2.putText(image, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
        cv2.putText(image, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 255, 255), 1)

    
    #cv2.imshow("input", image)

cap.release()
cv2.destroyAllWindows()
