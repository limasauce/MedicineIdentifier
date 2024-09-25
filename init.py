import cv2
import numpy as np
import pytesseract

def detect_imprint():
    image = cv2.imread()
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray

def detect_color():
    # Load image
    image = cv2.imread()
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    colors = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255]),
        'blue': ([100, 100, 100], [140, 255, 255]),
        # Add more colors as needed
    }

    detected_colors = []
    for color, (lower, upper) in colors.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        if cv2.countNonZero(mask) > 0:
            detected_colors.append(color)

    return detected_colors

def detect_input():
    print("Please enter the image you wish to enter:")
    img_path = input()
    return img_path


img_path = detect_input()
colors_found = detect_color(img_path)
image_gray = detect_imprint(img_path)

print("Detected Colors:", colors_found)
print(image_gray)
