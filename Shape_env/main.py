import cv2
import numpy as np

# Function to calculate angle between two vectors
def angle_between_vectors(v1, v2):
    v1 = v1.flatten()  # Flatten to 1D
    v2 = v2.flatten()  # Flatten to 1D
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return np.degrees(angle)

# reading image 
img = cv2.imread('C:\\Users\\DELL\\Desktop\\PyProject\\ShapeDetection\\Shape_env\\images\\shapes.jpg')

# converting image into grayscale image 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image 
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# using a findContours() function 
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

# list for storing names of shapes 
for contour in contours:

    # here we are ignoring first contour because 
    # findContours function detects whole image as shape 
    if i == 0:
        i = 1
        continue

    # cv2.approxPolyDP() function to approximate the shape
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # using drawContours() function (set color to green)
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 5)

    # finding center point of shape
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        # putting shape name at the center of each shape (set text color to green)
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif len(approx) == 4:
            # Further check for square vs rectangle or parallelogram
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Check if the shape is a square
            if aspect_ratio == 1:
                cv2.putText(img, 'Square', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Check if it's a rectangle or parallelogram by looking at the angles
                # Calculate the angles between the vectors of the sides of the quadrilateral
                angles = []
                for j in range(len(approx)):
                    pt1 = approx[j]
                    pt2 = approx[(j + 1) % len(approx)]
                    pt3 = approx[(j + 2) % len(approx)]

                    # Vectors for two consecutive edges
                    v1 = pt2 - pt1
                    v2 = pt3 - pt2
                    angle = angle_between_vectors(v1, v2)
                    angles.append(angle)

                # If any angle is not approximately 90 degrees, it's a parallelogram
                if all(np.isclose(angle, 90, atol=10) for angle in angles):
                    cv2.putText(img, 'Rectangle', (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'Parallelogram', (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            cv2.putText(img, 'Circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# displaying the image after drawing contours and text
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
