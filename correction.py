import cv2
import numpy as np
# from matplotlib import pyplot as plt

base_centroid = [503, 503]

def find_centroid(contour):
    # Finds the centroid of a contour #
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx,cy

def find_points_with_xMin(cont):
    min_x = 9999
    min_points = []
    for point in cont:
        if point[1] <= min_x:
            min_x = point[1]
            min_points.append(point)
    return min_points

def find_points_with_xMax (cont):

    max_x = 0
    max_points = []
    for point in cont:
        if point[1] >= max_x:
            max_x = point[1]
            max_points.append(point)
    return max_points

def find_points_with_yMax (cont):

    max_y = 0
    max_points = []
    for point in cont:
        if point[0] >= max_y:
            max_y = point[0]
            max_points.append(point)
    return max_points

def find_point_with_XMminYMax (cont):
    # Finding the top right point #
    min_points = find_points_with_xMin(cont)
    return find_points_with_yMax(min_points)[0]

def find_point_with_YMaxXMax (cont):
    # Finding the top right point #
    max_points = find_points_with_yMax(cont)
    return find_points_with_xMax(max_points)[0]

def find_angle(vec):
    #Find angle between the vector and the y-axis#
    y_axis = np.array([0, 1])

    # Calculate the dot product
    dot_product = np.dot(vec, y_axis)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vec)
    magnitude2 = np.linalg.norm(y_axis)

    # Calculate the cosine of the angle between the 2 lines
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in degrees
    return (np.arccos(cosine_angle) * 180 / np.pi)

def rotate_img_clockwise(img, angle):
    h,w = img.shape[:2]
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1)
    # Perform the rotation
    return (cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)))

def get_hough(img):
    # Find the Hough transfrom of the image #
    
    # Apply Canny edge detection 
    edges = cv2.Canny(img, 50, 150)
    # Apply Hough transfrom
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=190)
    # Draw Hough lines on a black mask and return it
    mask = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return mask

def get_contour(houghImg):
    # Find the exact and approximate contour of the qr code #

    # Find contours in the hough image
    contours, _ = cv2.findContours(houghImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours that have 4 edge points and max area
    max_area = 0
    selected_contour_exact = None
    selected_contour_appx = None

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                selected_contour_exact = contour
                selected_contour_appx = approx
    return (selected_contour_exact, selected_contour_appx)

def translate_to_middle(img, contour, tolerance_x=0, tolerance_y=0):
    # Translate the image to the middle of the frame #

    # Centroid of base image is right at the middle, Whiler Centroid of the #
    # current image is not, thus tolerance is added to correct this difference # 

    h,w = img.shape[:2]
    # Find img centroid
    old_centroid = find_centroid(contour=contour)
    # Find the translation matrix to the base centroid
    tx = base_centroid[0] - old_centroid[0] + tolerance_x
    ty = base_centroid[1] - old_centroid[1] + tolerance_y
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Translate the image
    return (cv2.warpAffine(img, translation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)))

def shift_perspective(img, contour, target_verticies):

    h,w = img.shape[:2]
    
    #   Original verticies from the contour
    bottom_left, top_left, top_right, bottom_right = contour.reshape(-1, 2)
    original_verticies = np.array([bottom_left, top_left, top_right, bottom_right], dtype= np.float32)

    # Apply prespective transfrom
    perspective_transform_matrix = cv2.getPerspectiveTransform(original_verticies, target_verticies)
    return (cv2.warpPerspective(img, perspective_transform_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)))

def rotation_correction(img):

    # Find Hough
    hough_lines = get_hough(img)
    # Find Contour
    exact,appx = get_contour(hough_lines)
    # # Draw contour
    # cont_img = img.copy()
    # if exact is not None:
    #     cv2.drawContours(cont_img, [exact], -1, (0, 0, 0), 4)
    # Find angle between contour side and y-axis 
    p1 = find_point_with_XMminYMax(exact.reshape(-1, 2))
    p2 = find_point_with_YMaxXMax(exact.reshape(-1, 2))
    vector = p2-p1
    angle = find_angle(vector) # 8.38 degrees
    # Rotate the image
    rotated_img = rotate_img_clockwise(img, angle)
    return rotated_img

def translation_correction(img):

    # Find Hough
    hough_lines = get_hough(img)
    # Find Contour
    exact,appx = get_contour(hough_lines)

    return translate_to_middle(img, exact, tolerance_x=0, tolerance_y=0)


img_1 = cv2.imread('images/06-solved.png', cv2.IMREAD_GRAYSCALE)
cont= rotation_correction(img_1)
# plt.imshow(cont, cmap='gray')
print(cont)