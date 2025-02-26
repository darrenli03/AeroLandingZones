import cv2
import numpy as np
import largestinteriorrectangle as lir 

#video from https://www.vecteezy.com/video/39396573-giraffes-running-in-the-desert-in-africa

# Convert RGB to HSV
target_rgb = np.uint8([[[159, 140, 131]]])
target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV)[0][0]

# print(target_hsv)

# Define the range for the target color in HSV, add value clamping to prevent invalid HSV ranges
hue = target_hsv[0]
sat = target_hsv[1]
val = target_hsv[2]

lower_color = np.array([
    max(0, hue - 20),
    max(0, sat - 40), 
    max(0, val - 40)
])

upper_color = np.array([
    min(179, hue + 20),
    min(255, sat + 40),
    min(255, val + 40)
])

# Function to process frame and detect areas of the target color
def detect_landing_areas(frame):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the target color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow('Mask', mask)

    # Find contours of the target color areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Define colors for the top 3 largest contours
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)] # Green, Blue, Red

    # Highlight the top 3 largest contours
    for i, contour in enumerate(contours[:3]):
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        # Draw the contour and its center on the frame
        cv2.drawContours(frame, [contour], -1, colors[i], 2)
        cv2.circle(frame, (cX, cY), 7, colors[i], -1)
        cv2.putText(frame, f"Landing Zone {i+1}", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        # Print the coordinates of the center of the largest contour
        if i == 0:
            print(f"Largest Landing Zone Center: ({cX}, {cY})")

    return frame


#switch between image and video testing
test = "video"
# test = "image"

if test == "video":
# Open the video file
    cap = cv2.VideoCapture('desertPOVedit.mp4')

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        
        # If the frame was read successfully, process it
        if ret:
            # Process the frame
            processed_frame = detect_landing_areas(frame)
            
            # Display the result
            cv2.imshow('Landing Area Detection', processed_frame)
            
            # Exit if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
elif test == "image":
    # Load the test image
    image = cv2.imread('vegetation.jpg')

    # Process the image
    processed_image = detect_landing_areas(image)

    # Display the result
    cv2.imshow('Landing Area Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()