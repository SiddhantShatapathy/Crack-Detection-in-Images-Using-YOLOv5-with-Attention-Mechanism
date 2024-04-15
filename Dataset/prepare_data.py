import os
import shutil
import random
import argparse

def create_datasets(main_dir, output_dir, split_ratio):
    # Path setup
    images_dir = os.path.join(main_dir, 'noisy_images')
    labels_dir = os.path.join(main_dir, 'labels')
    train_images_dir = os.path.join(output_dir, 'train/images')
    train_labels_dir = os.path.join(output_dir, 'train/labels')
    val_images_dir = os.path.join(output_dir, 'val/images')
    val_labels_dir = os.path.join(output_dir, 'val/labels')
    
    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Get list of all files and shuffle
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(image_files)  # Shuffle files to ensure random split

    # Determine split index
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Copy files to their respective directories
    for file in train_files:
        shutil.copy(os.path.join(images_dir, file), train_images_dir)
        corresponding_label = file.replace('.jpg', '.txt')  # Assuming image files are .jpg
        if os.path.exists(os.path.join(labels_dir, corresponding_label)):
            shutil.copy(os.path.join(labels_dir, corresponding_label), train_labels_dir)

    for file in val_files:
        shutil.copy(os.path.join(images_dir, file), val_images_dir)
        corresponding_label = file.replace('.jpg', '.txt')  # Assuming image files are .jpg
        if os.path.exists(os.path.join(labels_dir, corresponding_label)):
            shutil.copy(os.path.join(labels_dir, corresponding_label), val_labels_dir)

    print("Data has been shuffled and split into training and validation sets and copied to", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Shuffle and split datasets for YOLOv5 training.')
    parser.add_argument('--main_dir', type=str, help='Path to the main directory containing the "noisy_images" and "labels" folders.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory where the train and val folders will be created.')
    parser.add_argument('--split_ratio', type=float, help='Ratio of data to be used for training (e.g., 0.8 for 80% training data).')

    args = parser.parse_args()
    create_datasets(args.main_dir, args.output_dir, args.split_ratio)

if __name__ == '__main__':
    main()
