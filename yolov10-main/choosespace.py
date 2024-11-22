import cv2

# Define the path to the image to be processed
image_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\standard\Camera4_June_24_2024_09_51_00_41.png"

# Read the image
img = cv2.imread(image_path)
points = []

# Mouse callback function to get the coordinates of clicks
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click event
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a small green circle at the clicked position
        cv2.imshow("Image", img)

# Display the image and set the mouse callback
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", get_points)

print("Please click on the polygon vertices in the image, then press 'q' to exit and see the results.")

# Wait for the user to press the 'q' key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the selected vertices
print("Selected polygon vertices:", points)
