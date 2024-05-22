import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the frame
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Normalize the frame (make it fall within the range 0-1)
    normalized = cv2.normalize(blurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Create the colormap
    colormap = plt.get_cmap('hot')

    # Apply the colormap
    heatmap = colormap(normalized)

    # Remove the alpha channel
    heatmap = np.delete(heatmap, 3, 2)

    return heatmap

# Open the video source
cap = cv2.VideoCapture('video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # Apply the heatmap to the frame
        heatmap = create_heatmap(frame)

        # Display the resulting frame
        cv2.imshow('Heatmap', heatmap)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()