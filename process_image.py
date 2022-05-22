import cv2 as cv
import numpy as np



def preprocess_image(image):
    gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    threshold_image = cv.adaptiveThreshold(cv.GaussianBlur(gray_scale, (3,3),6),255,1,1,11,2)
    return threshold_image

def contouring(image, threshold_image):
    contour_1 = image.copy()
    contour_2 = image.copy()
    contour, hierarchy = cv.findContours(threshold_image ,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_color = (0,255,0)
    cv.drawContours(contour_1, contour,-1,contour_color,3)

    return contour_1, contour_2, contour, hierarchy

def outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv.contourArea(i)
        if area >50:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area

def reframe_image(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes


def wrap(image, contour, contour_fig2):
  black_img = np.zeros((450,450,3), np.uint8)
  biggest, maxArea = outline(contour)
  if biggest.size != 0:
      biggest = reframe_image(biggest)
      cv.drawContours(contour_fig2,biggest,-1, (0,255,0),10)
      pts1 = np.float32(biggest)
      pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
      matrix = cv.getPerspectiveTransform(pts1,pts2)
  imagewrap = cv.warpPerspective(image,matrix,(450,450))
  imagewrap =cv.cvtColor(imagewrap, cv.COLOR_BGR2GRAY)

  return imagewrap

def crop(cells):
    cropped = []
    for i in cells:
        cropped.append(np.array(i)[4:46, 6:46])
    return cropped