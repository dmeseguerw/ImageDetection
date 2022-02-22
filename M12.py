import cv2
import numpy as np
import os
from tkinter import Tk
import math
import argparse
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)


def filter_distance(mtc):
    dist = [m.distance for m in mtc]
    thres_dist = (sum(dist) / len(dist)) * 0.5

    # keep only the reasonable matches
    sel_matches = [m for m in mtc if m.distance < thres_dist]
    # print ('#selected matches:%d (out of %d)' % (len(sel_matches), len(mtc)))
    return sel_matches


def check_point(pt1,pts,rad):
    print(pt1)
    print(pts)
    if (pt1[0] >= pts[0][0] and pt1[1] >= pts[0][1] and pt1[0] <= pts[1][0] and pt1[1] <= pts[1][1]):
        print("TRUEEEE")
        return True
    else:
        return False


def check_matches(list1,pts,rad):
    trueCtr = 0

    for i in list1:
        res = check_point(i,pts,rad)
        # print(res)
        if(res):
            trueCtr = trueCtr + 1
            print("SDFDSFSDFSDFSDF")
    # print("Length of list: ", len(list1))
    # print("Length of counter: ", trueCtr)
    if trueCtr >= 0.8*len(list1): return True
    else: return False


def get_middle(list):
    avX = 0
    avY = 0
    for i in range(len(list)):
        avX += list[i][0]
        avY += list[i][1]
    avX = avX / len(list)
    avY = avY / len(list)
    return avX, avY


def get_longest(pt, list):
    distance = 0
    longestD = 0
    longestX = 0
    longestY = 0
    for i in list:
        distance = math.sqrt((pt[0] - i[0]) ** 2 + (pt[1] - i[1]) ** 2)
        if (longestD < distance):
            longestD = distance

    return longestD


def rect_center(ref_point, center):
    x = (abs(ref_point[1][0]-ref_point[0][0])/2)
    y = (abs(ref_point[1][1] - ref_point[0][1])/2)
    new_ref_point = [(int(center[0] - x),int(center[1] - y)),(int(center[0] + x),int(center[1] + y))]
    return new_ref_point


def analyze_all(fpath,sel, ref_point):
    templates = []
    img_analyzed = input("Ingrese el nombre del patron a buscar: ")

    for dirName, subDirs, files in fpath:
        for f in files:
            templates.append('Imag/' + f)
            # print(f)

    for image in templates:
        im2 = cv2.imread('Database/' + img_analyzed)
        im1 = cv2.imread(image)

        # img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        img1 = im1
        img2 = im2
        orb = cv2.ORB_create(
            edgeThreshold=7, patchSize=20, nlevels=8,
            fastThreshold=7, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
            firstLevel=0, nfeatures=1000)  # initialize ORB object
        kp1, des1 = orb.detectAndCompute(img1, None)  # Get keypoints and descriptors for image
        kp2, des2 = orb.detectAndCompute(img2, None)  # Get keypoints and descriptors for template image
        pts1 = cv2.KeyPoint_convert(kp1)
        pts2 = cv2.KeyPoint_convert(kp2)

        # Create matches and sort them
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)
        if len(matches) > 25:
            matches = sorted(matches, key=lambda x: x.distance)  # sorted matches
            matches = filter_distance(matches)
        if len(matches) > 25:
            list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]  # coordinates of all matches in img1
            list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]  # coordinates of all matches in img2
            x1, y1 = get_middle(list_kp1)
            x2, y2 = get_middle(list_kp2)

            # longestDistance = get_longest([x2,y2],list_kp2) ********** TO BE DONE
            ref_point_aux = rect_center(ref_point, [x1, y1])
            val = check_matches(list_kp1, ref_point_aux, 80)  # Check if it's a real match
            # print("MATCH: ", val)
            if val:
                # Draw Keypoints
                if sel == '2':
                    img3 = cv2.drawKeypoints(img1, kp1, None, flags=None)
                    img4 = cv2.drawKeypoints(img2, kp2, None, flags=None)
                    cv2.imshow("Keypoints image", img3)
                    cv2.imshow("Keypoints template", img4)
                    cv2.waitKey(0)

                    # img1_circle = img1
                    # img2_circle = img2
                    img_rect1 = img1
                    img_rect2 = img2
                    # img1_circle = cv2.circle(img1_circle, (int(x1), int(y1)), 80, (0, 0, 255), 2)
                    # img2_circle = cv2.circle(img2_circle, (int(x2), int(y2)), 80, (0, 0, 255), 2)
                    print(ref_point)
                    print(ref_point_aux)
                    img_rect2 = cv2.rectangle(img_rect2, ref_point[0], ref_point[1], (0, 255, 0), 2)
                    img_rect1 = cv2.rectangle(img_rect1, ref_point_aux[0], ref_point_aux[1], (0, 255, 0), 2)
                    # cv2.imshow("Keypoint matches", img_rect1)
                    # cv2.imshow("Keypoint matches", img_rect2)
                    # cv2.waitKey(0)

                    # Show matches:
                    img5 = cv2.drawMatches(img_rect1, kp1, img_rect2, kp2, matches[:len(kp2) - 1], None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow("Keypoint matches", img5)
                    cv2.waitKey(0)
            else:
                os.remove(image)
                print("Removed out of area " + image)
        else:
            os.remove(image)
            print("Removed no matches" + image)


def select_option(ref_point):
    print("Seleccione una opcion: \n1. Buscar patron. \n2. Visualizar resultdos \n3. Finalizar \n")
    sel = input()
    path = os.walk("Imag")
    analyze_all(path,sel, ref_point)

    return sel


def get_size(img):
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    return height, width


def divide_image(img):
    for r in range(600, img.shape[0] - 1200, 650):
        for c in range(650, img.shape[1] - 1100, 650):
            cv2.imwrite("Imag/" + f"img{r}_{c}.png", img[r:r + 650, c:c + 650, :])



# now let's initialize the list of reference point
ref_point = []

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)


# keep looping until the 'c' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # press 'r' to reset the window
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# close all open windows
cv2.destroyAllWindows()


im1 = cv2.imread('Images/UVF_MI_FinalJPEG.jpg')
divide_image(im1)
sel = 0
while(sel != '3'):
    select_option(ref_point)