import datetime
import os
import shutil
import sys
import cv2
import numpy as np
import pandas as pd  # 导入 pandas 库
from fpdf import FPDF
from ultralytics import YOLO
from PIL import Image
from PyPDF2 import PdfMerger

# Adding the custom library path for importing the process_image function
sys.path.append(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main')
from OpenCV import process_image  # Assuming the function to process images is called process_image

def convert_to_png(img_path):
    file_name, file_extension = os.path.splitext(img_path)
    if file_extension.lower() != '.png':
        img = Image.open(img_path)
        new_img_path = file_name + '.png'
        img.save(new_img_path, 'PNG')
        print(f"Image converted to PNG: {new_img_path}")
        return new_img_path
    else:
        img = Image.open(img_path)
        img.save(img_path, 'PNG')
        return img_path

def process_image_and_generate_report(img_url, pdf_file_list, excel_data):
    try:
        # Ensure the image is in PNG format
        img_url = convert_to_png(img_url)

        # Generate the current time string for file naming
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define the path to save the temporary PDF file for each image
        pdf_temp_path = rf"C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\result_{current_time_str}.pdf"

        # Load different models and set thresholds based on the image name
        if 'camera1' in img_url.lower():
            model = YOLO(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect\train\weights\best.pt')
            roi_points = [(299, 92), (305, 78), (316, 68), (335, 77), (363, 85),
                          (369, 100), (371, 118), (338, 116), (319, 108), (309, 100)]
            confidence_threshold = 0.7
            area_ratio_threshold = 0.1
        elif 'camera4' in img_url.lower():
            model = YOLO(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect\train\weights\best.pt')
            roi_points = [(304, 56), (308, 130), (300, 180), (305, 253), (302, 311),
                          (313, 343), (355, 342), (410, 352), (454, 350), (533, 353),
                          (600, 351), (637, 344), (628, 282), (628, 235), (629, 190),
                          (629, 117), (628, 80), (627, 43), (606, 27), (547, 17),
                          (491, 26), (422, 22), (360, 42), (313, 45)]
            confidence_threshold = 0.3
            area_ratio_threshold = 0.01
        else:
            raise ValueError("Image path does not contain 'camera1' or 'camera4', unable to determine reference image and ROI.")

        # Process the image
        processed_img = process_image(img_url)
        save_dir = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\processed"
        os.makedirs(save_dir, exist_ok=True)
        processed_img_path = os.path.join(save_dir, "processed_image.png")
        cv2.imwrite(processed_img_path, processed_img)

        # Perform inference on the processed image and save the results
        results = model(processed_img_path, save=True)

        # Initialize the PDF object
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(200, 10, txt=f"Current Time: {current_time}", ln=True)
        pdf.cell(200, 10, txt="Original Image:", ln=True)
        pdf.image(img_url, x=10, y=30, w=90)

        output_texts = []
        retrain_required = False
        has_overlap = False
        area_ratio = 0

        # Process inference results with dynamic confidence threshold
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box output
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()  # Confidence
                    if conf > confidence_threshold:
                        retrain_required = True
                        break

        # If there is a box with confidence greater than threshold, reload the segmentation model
        segmentation_image_path = None
        if retrain_required:
            print(f"Confidence greater than {confidence_threshold}, reloading the segmentation model for training...")
            model_seg = YOLO(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\segment\train\weights\best.pt')

            # Perform inference on the image again
            results_seg = model_seg(processed_img_path, save=True)

            # Get the path of the saved segmentation image
            segmentation_image_path = os.path.join(results_seg[0].save_dir, os.path.basename(processed_img_path))

            roi_contour = np.array(roi_points, dtype=np.int32)
            total_image_area = None
            has_overlap = False

            for result in results_seg:
                masks = result.masks
                if masks is not None:
                    total_image_area = result.orig_img.shape[0] * result.orig_img.shape[1]

                    for i in range(len(masks.data)):
                        conf = result.boxes[i].conf.item()
                        mask = masks.data[i].cpu().numpy()
                        mask_area = np.sum(mask > 0)

                        # Create a mask for the ROI area
                        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
                        cv2.fillPoly(roi_mask, [roi_contour], 1)

                        # Calculate the overlap between the ROI and the detected mask
                        overlap = np.logical_and(mask, roi_mask)
                        overlap_area = np.sum(overlap)

                        # Check for overlap
                        if overlap_area > 0:
                            has_overlap = True
                            output_texts.append("Some fallen leaves near the drain, please clean up immediately!")
                            break

                        # Save data to excel_data
                        excel_data.append({
                            'Image Name': os.path.basename(img_url),
                            'Confidence': conf,
                            'Area Ratio': mask_area / total_image_area,
                            'Overlap': has_overlap
                        })

                else:
                    print(f"Error: No masks data found for image: {img_url}")
                    output_texts.append("Very clean, no need to clean up.")
                    break

            # Evaluate high-confidence detections and their area ratio
            if not has_overlap:
                any_high_confidence = False
                for i in range(len(masks.data)):
                    conf = result.boxes[i].conf.item()
                    if conf > confidence_threshold:
                        any_high_confidence = True
                        mask_area = np.sum(masks.data[i].cpu().numpy() > 0)
                        area_ratio = mask_area / total_image_area

                        # Apply different area_ratio thresholds based on camera type
                        if area_ratio > area_ratio_threshold:
                            output_texts.append("Large pile of fallen leaves, please clean up immediately!")
                        else:
                            output_texts.append("There is some accumulation of fallen leaves, but no need to clean up.")
                        break

                if not any_high_confidence:
                    output_texts.append("The current environment is relatively clean, no cleaning is needed!")

            # Add the images to the PDF report
            pdf.cell(200, 10, txt="Original Image and Segmentation Result:", ln=True)
            pdf.image(img_url, x=10, y=30, w=90)
            if segmentation_image_path and os.path.exists(segmentation_image_path):
                pdf.image(segmentation_image_path, x=110, y=30, w=90)
        else:
            output_texts.append("Very clean, no need to clean up.")

        # Add the inference results to the PDF
        pdf.set_y(90)
        pdf.cell(200, 10, txt="Inference Results:", ln=True)
        for text in output_texts:
            pdf.cell(200, 10, txt=text, ln=True)

        # Save the PDF report
        pdf.output(pdf_temp_path)
        print(f"Results have been saved to {pdf_temp_path}")

        # Add the generated PDF to the list for merging
        pdf_file_list.append(pdf_temp_path)

    except AttributeError as e:
        # Handle any errors and create a PDF with default output
        print(f"Error encountered: {e}. Image: {img_url}")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(200, 10, txt=f"Current Time: {current_time}", ln=True)
        pdf.cell(200, 10, txt="Original Image:", ln=True)
        pdf.image(img_url, x=10, y=30, w=90)
        pdf.set_y(90)
        pdf.cell(200, 10, txt="Inference Results:", ln=True)
        pdf.cell(200, 10, txt="Very clean, no need to clean up.", ln=True)

        pdf.output(pdf_temp_path)
        print(f"Results for error case have been saved to {pdf_temp_path}")

        # Add the generated PDF to the list for merging
        pdf_file_list.append(pdf_temp_path)

def merge_pdfs(pdf_file_list, output_pdf_path):
    merger = PdfMerger()
    for pdf_file in pdf_file_list:
        merger.append(pdf_file)
    merger.write(output_pdf_path)
    merger.close()
    print(f"Merged PDF saved at: {output_pdf_path}")

    for pdf_file in pdf_file_list:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
            print(f"Deleted temporary PDF: {pdf_file}")

def delete_predict_folders():
    detect_dir = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect"
    segment_dir = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\segment"

    def delete_folders_with_predict(base_dir):
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path) and "predict" in folder_name:
                print(f"Deleting folder: {folder_path}")
                try:
                    os.rmdir(folder_path)
                except OSError:
                    shutil.rmtree(folder_path)

    delete_folders_with_predict(detect_dir)
    delete_folders_with_predict(segment_dir)

def main():
    img_dir = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\test'
    pdf_file_list = []
    excel_data = []

    # Process each image and generate individual PDFs
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_and_generate_report(img_path, pdf_file_list, excel_data)

    # Merge all generated PDFs into one
    output_pdf_path = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\merged_results.pdf'
    merge_pdfs(pdf_file_list, output_pdf_path)

    # Delete the temporary predict folders
    delete_predict_folders()

    # Save Excel data
    excel_df = pd.DataFrame(excel_data)
    excel_output_path = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\inference_results.xlsx'
    excel_df.to_excel(excel_output_path, index=False)
    print(f"Excel file saved at: {excel_output_path}")

if __name__ == "__main__":
    main()
