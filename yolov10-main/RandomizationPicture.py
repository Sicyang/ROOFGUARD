import os
import random
import shutil

# Define the paths
source_dir = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\train'
test_dir = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\test'
val_dir = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\val'

# Create target directories if they do not exist
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all files in the source directory
all_files = os.listdir(source_dir)
total_files = len(all_files)

# Calculate the number of files for each subset
num_test = int(total_files * 0.2)
num_val = num_test  # 20% of the files go to the validation set
num_train = total_files - num_test - num_val  # The remaining files go to the training set (60%)

# Shuffle the files randomly
random.shuffle(all_files)

# Split the files into train, test, and validation sets
test_files = all_files[:num_test]
val_files = all_files[num_test:num_test + num_val]
train_files = all_files[num_test + num_val:]

# Move files from the source directory to the target directories
for file_name in test_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(test_dir, file_name))

for file_name in val_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(val_dir, file_name))

# train_files remain in the source directory
print(f"File distribution complete!\nTest set: {len(test_files)} images\nValidation set: {len(val_files)} images\nTraining set: {len(train_files)} images")
