import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

def prepare_dataset(data_dir, output_dir):
    """
    Reads TIF images from the talisman test suite directory, 
    extracts labels from their filenames, and saves them as 
    train and test .npy arrays to prevent data leakage.
    """
    
    train_images, train_labels = [], []
    test_images, test_labels   = [], []
    
    # Define our class mapping based on the README
    class_mapping = {
        'healthy': 0,
        'emphysema': 1,
        'ground_glass': 2,
        'fibrosis': 3,
        'micronodules': 4
    }
    
    print(f"Using class mapping: {class_mapping}")
    
    # Get all .tif files in the directory
    image_paths = glob.glob(os.path.join(data_dir, '*.tif'))
    
    if not image_paths:
        print(f"No .tif files found in {data_dir}.")
        return
        
    print(f"Found {len(image_paths)} images. Processing train/test splits...")
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(img_path)
        
        # Determine if this is a training patch (contains 'patient-1_')
        is_train = 'patient-1_' in filename
        
        # Extract the class name from the filename
        class_name = filename.split('_')[0]
        if class_name == 'ground':
            class_name = 'ground_glass'
            
        if class_name not in class_mapping:
            print(f"Warning: Unknown class '{class_name}' for file {filename}. Skipping.")
            continue
            
        class_idx = class_mapping[class_name]
        
        # Read TIF image (Hounsfield Units)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Failed to read {filename}. Skipping.")
            continue
            
        if img.shape[:2] != (32, 32):
             img = cv2.resize(img, (32, 32))
            
        if is_train:
            train_images.append(img)
            train_labels.append(class_idx)
        else:
            test_images.append(img)
            test_labels.append(class_idx)
            
    # Convert to numpy arrays
    tr_imgs_np = np.array(train_images)
    tr_lbls_np = np.array(train_labels)
    te_imgs_np = np.array(test_images)
    te_lbls_np = np.array(test_labels)
    
    print(f"\n--- Train Set ---")
    print(f"Images: {tr_imgs_np.shape}, Labels: {tr_lbls_np.shape}")
    tr_unique, tr_counts = np.unique(tr_lbls_np, return_counts=True)
    for c, count in zip(tr_unique, tr_counts):
        name = [k for k, v in class_mapping.items() if v == c][0]
        print(f"  {name}: {count}")

    print(f"\n--- Test Set (LOPO) ---")
    print(f"Images: {te_imgs_np.shape}, Labels: {te_lbls_np.shape}")
    te_unique, te_counts = np.unique(te_lbls_np, return_counts=True)
    for c, count in zip(te_unique, te_counts):
        name = [k for k, v in class_mapping.items() if v == c][0]
        print(f"  {name}: {count}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving to .npy files...")
    np.save(os.path.join(output_dir, 'train_images.npy'), tr_imgs_np)
    np.save(os.path.join(output_dir, 'train_labels.npy'), tr_lbls_np)
    np.save(os.path.join(output_dir, 'test_images.npy'), te_imgs_np)
    np.save(os.path.join(output_dir, 'test_labels.npy'), te_lbls_np)
    
    print("Done! Data leakage issue resolved.")
    
if __name__ == "__main__":
    input_dir = 'ILD_DB/ILD_DB_talismanTestSuite'
    output_dir = 'ILD_DB_npy'
    
    prepare_dataset(input_dir, output_dir)
