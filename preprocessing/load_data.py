# Paths to LEVIR-CD+ dataset
image1_train_dir = "/kaggle/input/levir-cd/LEVIR CD/train/A"
image2_train_dir = "/kaggle/input/levir-cd/LEVIR CD/train/B"
mask_train_dir = "/kaggle/input/levir-cd/LEVIR CD/train/label"

image1_test_dir = "/kaggle/input/levir-cd/LEVIR CD/test/A"
image2_test_dir = "/kaggle/input/levir-cd/LEVIR CD/test/B"
mask_test_dir = "/kaggle/input/levir-cd/LEVIR CD/test/label"

input_shape = (256, 256)  # Resize dimensions

def load_images(image1_dir, image2_dir, mask_dir):
    image1_files = sorted(os.listdir(image1_dir))
    image2_files = sorted(os.listdir(image2_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    X = []
    y = []
    
    for img1, img2, mask in zip(image1_files, image2_files, mask_files):
        
        img1_path = os.path.join(image1_dir, img1)
        img2_path = os.path.join(image2_dir, img2)
        mask_path = os.path.join(mask_dir, mask)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize images and mask
        img1 = cv2.resize(img1, input_shape)
        img2 = cv2.resize(img2, input_shape)
        mask = cv2.resize(mask, input_shape)
        
        # Normalize images and mask
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        mask = mask / 255.0
        
        # Stack images along the channel axis
        stacked_image = np.concatenate([img1, img2], axis=-1)  # Shape: (256, 256, 6)
        
        X.append(stacked_image)
        y.append(mask)
    
    return np.array(X), np.array(y)

# Load the dataset
X, y = load_images(image1_train_dir, image2_train_dir, mask_train_dir)
X_test, y_test = load_images(image1_test_dir, image2_test_dir, mask_test_dir)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)