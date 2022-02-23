import cv2
import numpy as np
import os
from tkinter import *
import math
import argparse
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)


# Method used to keep only reasonable matches
def filter_distance(mtc):
    dist = [m.distance for m in mtc]
    thres_dist = (sum(dist) / len(dist)) * 0.5

    # keep only the reasonable matches
    sel_matches = [m for m in mtc if m.distance < thres_dist]
    return sel_matches

# Method used to check if point is a match
def check_point(pt1,pts,rad):
    if (pt1[0] >= pts[0][0] and pt1[1] >= pts[0][1] and pt1[0] <= pts[1][0] and pt1[1] <= pts[1][1]):
        return True
    else:
        return False

# Method to print accuracy data
def print_data(matchesFound, totalMatches):
    summaryFile = open("summaryFile.txt", "a")
    print("----------------------------------")
    summaryFile.write("---------------------------------- \n")
    print("Matches Found: " + str(matchesFound))
    summaryFile.write("Matches Found: " + str(matchesFound) + "\n")
    print("Total Matches: " + str(totalMatches))
    summaryFile.write("Total Matches: " + str(totalMatches) + "\n")
    print("Accuracy rate: " + str(round(((matchesFound/totalMatches)*100),2)) + "%")
    summaryFile.write("Accuracy rate: " + str(round(((matchesFound/totalMatches)*100),2)) + "%" + "\n")
    summaryFile.close()

# Method used to check if points are matches in 80% of points
def check_matches(list1,pts,rad):
    trueCtr = 0
    for i in list1:
        res = check_point(i,pts,rad)
        if(res):
            trueCtr = trueCtr + 1
        # print_data(trueCtr, len(list1))
    if trueCtr >= 0.9*len(list1): 
        print_data(trueCtr, len(list1))
        return True
    else: 
        return False

# Method used to get average of points
def get_middle(list):
    avX = 0
    avY = 0
    for i in range(len(list)):
        avX += list[i][0]
        avY += list[i][1]
    avX = avX / len(list)
    avY = avY / len(list)
    return avX, avY

# Method used to get far most point
def get_longest(pt, list):
    distance = 0
    longestD = 0
    for i in list:
        distance = math.sqrt((pt[0] - i[0]) ** 2 + (pt[1] - i[1]) ** 2)
        if (longestD < distance):
            longestD = distance

    return longestD

# Method used to get center of rectangle in segment image
def rect_center(reference, center):
    x = (abs(reference[1][0] - reference[0][0])/2)
    y = (abs(reference[1][1] - reference[0][1])/2)
    new_ref_point = [(int(center[0] - x),int(center[1] - y)),(int(center[0] + x),int(center[1] + y))]
    return new_ref_point

# Method used to perform all analysis around image
def analyze_all(fpath,sel):
    templates = []
    global patternImage
    # img_analyzed = input("Ingrese el nombre del patron a buscar: ")
    img_analyzed = patternImage
    im2 = cv2.imread(img_analyzed)
    crop = im2[ref_point[0][1]:ref_point[1][1],ref_point[0][0]:ref_point[1][0]]
    img_analyzed = patternImage.replace('.JPG','_cropped.JPG')
    cv2.imwrite(img_analyzed,crop)
    cropped = cv2.imread(img_analyzed)
    # cv2.imshow('Database/test.png', im2[ref_point[1][1]:ref_point[0][1], ref_point[1][0]:ref_point[0][0]])
    for dirName, subDirs, files in fpath:
        for f in files:
            templates.append('Segments/' + f)
    img2 = cropped
    orb = cv2.ORB_create(
            edgeThreshold=7, patchSize=20, nlevels=8,
            fastThreshold=7, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
            firstLevel=0, nfeatures=1000)  # initialize ORB object
    kp2, des2 = orb.detectAndCompute(img2, None)  # Get keypoints and descriptors for template image
    for image in templates:
        im1 = cv2.imread(image)

        # img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        img1 = im1

        kp1, des1 = orb.detectAndCompute(img1, None)  # Get keypoints and descriptors for image

        # Create matches and sort them
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(des1, des2, None)
        if len(matches) > 25:
            matches = sorted(matches, key=lambda x: x.distance)  # sorted matches
            matches = filter_distance(matches)
        if len(matches) > 25:
            list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]  # coordinates of all matches in segment
            x1, y1 = get_middle(list_kp1)
            ref_point_aux = rect_center(ref_point, [x1, y1])
            realMatch = check_matches(list_kp1, ref_point_aux, 80)  # Check if it's a real match
            if realMatch:
                # Draw Keypoints
                print("REAL MATCH" + str(len(matches)))
                img3 = cv2.drawKeypoints(img1, kp1, None, flags=None)
                img4 = cv2.drawKeypoints(img2, kp2, None, flags=None)
                # cv2.imshow("Keypoints image", img3)
                # cv2.imshow("Keypoints template", img4)
                # cv2.waitKey(0)

                img_rect1 = img1
                img_rect2 = img2
                # print(ref_point)
                # print(ref_point_aux)
                # img_rect2 = cv2.rectangle(img_rect2, ref_point[0], ref_point[1], (0, 255, 0), 2)
                img_rect1 = cv2.rectangle(img_rect1, ref_point_aux[0], ref_point_aux[1], (0, 255, 0), 2)

                # Show matches:
                img5 = cv2.drawMatches(img_rect1, kp1, cropped, kp2, matches[:len(kp2) - 1], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite("Matches/matches_" + str(image[9:]), img5)
                print("Matches/matches_" + str(image[9:]))
                sumFile = open("summaryFile.txt", "a")
                sumFile.write("Matches/matches_" + str(image[9:]) + "\n")
                sumFile.close()
                # cv2.waitKey(0)
            # else:
                # os.remove(image)
                # print("Removed out of area " + image)
        # else:
            # os.remove(image)
            # print("Removed no matches" + image)

# Method for menu
def select_option():
    print("-------------------------------------------------\nSeleccione una opcion:\n1. Seleccionar patron \n2. Segmentar imagen. \n3. Obtener resultados \n4. Finalizar \n")
    sel = input()
    path = os.walk("Segments")
    global image,clone,patternImage
    if(sel=="1"):
        patternImage = "Database/" + input("Seleccione una imagen en la base de datos:")
        image = cv2.imread(patternImage)
        clone = image.copy()
        print("Seleccionando patron...")
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
        select_option()
    if(sel=="2"):
        print("Segmentando imagen...")
        im1 = cv2.imread('Images/UVF_MI_FinalJPEG.jpg')
        divide_image(im1)
        select_option()
    if(sel=="4"):
        print("Aplicacion cerrada")
        exit()
    if(sel=="3"):
        print("Obteniendo resultados...")
        sumFile = open('summaryFile.txt','a')
        sumFile.truncate(0)
        sumFile.close()
        print(ref_point)
        analyze_all(path,sel)
        select_option()
    return sel

# Method used to retrieve image size
def get_size(img):
    height = img.shape[0]
    width = img.shape[1]
    return height, width

# Method used to divide image into segments
def divide_image(img):
    for r in range(0, img.shape[0] - 750, 650):
        for c in range(0, img.shape[1] - 750, 650):
            cv2.imwrite("Segments/" + f"img{r}x{r+750}_{c}x{c+750}.png", img[r:r + 750, c:c + 750, :])

# now let's initialize the list of reference point
ref_point = []

# Method used to draw rectangle around pattern image
def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        print("saving1 ")
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        print("saving2 ")
        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())


# load the image, clone it, and setup the mouse callback function
image = [[0]*2 for _ in range(3)]
patternImage = ""
clone = image.copy()
select_option()

