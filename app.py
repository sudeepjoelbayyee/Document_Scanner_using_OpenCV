import cv2
import numpy as np

width_img = 480
height_img = 720

# Initialize webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, width_img)  # Set width
cap.set(4, height_img)  # Set height
cap.set(10, 150)  # Set brightness


def pre_processing(img):
    img_gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1)
    img_canny = cv2.Canny(img_blur, threshold1=180, threshold2=200)
    kernel = np.ones((5, 5))
    img_dilation = cv2.dilate(img_canny, kernel=kernel, iterations=2)
    img_threshold = cv2.erode(img_dilation, kernel=kernel, iterations=1)
    return img_threshold


def getContours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    if biggest.size != 0:
        cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def get_wrap(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width_img, height_img))

    img_cropped = img_output[20: img_output.shape[0] - 20, 20:img_output.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (width_img, height_img))

    return img_cropped


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgContour = img.copy()
    img_threshold = pre_processing(img)
    biggest = getContours(img_threshold)

    if biggest.size != 0:
        img_warped = get_wrap(img, biggest)
    else:
        img_warped = imgContour.copy()  # Return the original image if no contour is found

    img_array = ([img, img_threshold],
                 [imgContour, img_warped])
    stacked_images = stackImages(0.6, img_array)
    cv2.imshow("Workflow", stacked_images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
