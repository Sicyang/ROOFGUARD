import cv2
import numpy as np
import os

def process_image(image_path):
    """
    Process the image: resize it, convert to 8-bit, extract the region of interest (ROI),
    and apply histogram equalization to standardize brightness.
    """
    print(f"Processing image: {image_path}")

    def convert_to_8bit(image):
        """
        Convert the image to 8-bit if it is not already in that format.
        """
        if image.dtype == np.uint16:  # Check if the image is 16-bit
            image = (image / 256).astype('uint8')  # Convert from 16-bit to 8-bit
        elif image.dtype != np.uint8:  # Check if the image is not already 8-bit
            image = image.astype('uint8')
        return image

    # Determine the reference image path and ROI points based on the input image name
    if 'camera4' in image_path.lower():
        reference_image_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\standard\Camera4_June_24_2024_09_51_00_41.png"
        roi_points = [(528, 134), (520, 208), (514, 291), (506, 367), (503, 440), (484, 536), (484, 576),
                      (517, 614), (568, 637), (624, 649), (735, 679), (760, 700), (815, 708), (878, 715),
                      (942, 718), (1002, 711), (1072, 712), (1153, 714), (1198, 712), (1251, 710), (1277, 711),
                      (1269, 643), (1268, 592), (1268, 521), (1276, 481), (1274, 432), (1272, 397), (1272, 333),
                      (1273, 274), (1278, 201), (1273, 146), (1272, 74), (1245, 26), (1171, 16), (1101, 11),
                      (1009, 16), (937, 10), (855, 17), (783, 22), (689, 50), (608, 80), (538, 108), (575, 97)]

    elif 'camera1' in image_path.lower():
        reference_image_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\standard\Camera1_April_12_2024_11_35_00_43.png"
        roi_points = [(83, 358), (94, 337), (113, 312), (138, 273), (161, 235), (178, 212), (201, 176),
                      (219, 153), (243, 125), (267, 97), (288, 78), (308, 62), (322, 52), (331, 63),
                      (350, 73), (367, 79), (374, 99), (383, 133), (390, 165), (396, 185), (404, 219),
                      (416, 253), (425, 290), (432, 320), (441, 359)]
    else:
        raise ValueError("Image path does not contain 'camera1' or 'camera4', unable to determine reference image and ROI.")

    # Read reference image
    reference_img = cv2.imread(reference_image_path)
    reference_img = convert_to_8bit(reference_img)
    ref_height, ref_width = reference_img.shape[:2]  # Get the reference image size

    # Create a mask with the same size as the reference image
    mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points)], 255)  # Fill the polygon area with white (255)

    # Read and process the input image
    img = cv2.imread(image_path)
    img = convert_to_8bit(img)

    # Resize the image to the reference image size
    resized_img = cv2.resize(img, (ref_width, ref_height))

    # Create a blank image to save the region of interest
    roi_img = np.zeros_like(resized_img)

    # Retain only the region of interest
    roi_img[mask == 255] = resized_img[mask == 255]

    # Convert to grayscale for histogram equalization
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to the ROI
    equalized_roi = cv2.equalizeHist(gray_roi)

    # Convert the equalized image back to BGR format for saving
    final_img = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)

    # Use the equalized image in the region of interest
    roi_img[mask == 255] = final_img[mask == 255]

    return roi_img


# def main(folder_path):
#     """
#     Process all images in the folder, save processed images with '_cv' suffix, and delete the original images.
#     """
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.png') or filename.endswith('.jpg'):
#             image_path = os.path.join(folder_path, filename)
#
#             # Process the image
#             processed_image = process_image(image_path)
#
#             # Save the processed image with the '_cv' suffix
#             new_filename = filename.split('.')[0] + '_cv.' + filename.split('.')[1]
#             new_image_path = os.path.join(folder_path, new_filename)
#             cv2.imwrite(new_image_path, processed_image)
#
#             # Delete the original image
#             os.remove(image_path)
#             print(f"Processed and deleted: {filename}")
#
#
# if __name__ == '__main__':
#     folder_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\val"
#
#     main(folder_path)
