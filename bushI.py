import cv2
import numpy as np

def detect_obstructions(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a binary image
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the obstructions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Highlight the obstructions
    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Green color for obstructions
    
    # Divide the frame into 16 sections (4x4)
    height, width = frame.shape[:2]
    section_height = height // 4
    section_width = width // 4
    
    min_density = float('inf')
    best_section = None
    
    # Calculate the density of obstructions in each section
    for i in range(4):
        for j in range(4):
            section_mask = mask[i * section_height:(i + 1) * section_height, j * section_width:(j + 1) * section_width]
            density = cv2.countNonZero(section_mask) / (section_height * section_width)
            
            if density < min_density:
                min_density = density
                best_section = (i, j)
    
    # Highlight the section with the least density of obstructions
    if best_section is not None:
        i, j = best_section
        cv2.rectangle(frame, (j * section_width, i * section_height), 
                      ((j + 1) * section_width, (i + 1) * section_height), 
                      (255, 0, 0), 2)  # Blue color for the best section
    
    return frame

# Switch between image and video testing
test = "video"
# test = "image"

if test == "video":
    # Open the video file
    cap = cv2.VideoCapture('desertPOVedit.mp4')
    paused = False

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while cap.isOpened():
        if not paused:
            # Read a frame from the video
            ret, frame = cap.read()
        
            # If the frame was read successfully, process it
            if ret:
                # Process the frame
                processed_frame = detect_obstructions(frame)
                
                # Display the result
                cv2.imshow('Obstruction Detection', processed_frame)
            else:
                break
        
        # Check for key presses
        key = cv2.waitKey(30) & 0xFF  # Adjust the delay to control playback speed (e.g., 30 for slower playback)
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused  # Toggle pause/play

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
elif test == "image":
    # Load the test image
    image = cv2.imread('vegetation.jpg')

    # Process the image
    processed_image = detect_obstructions(image)

    # Display the result
    cv2.imshow('Obstruction Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()