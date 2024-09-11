MRI Brain Image Classification with Pretrained 3D CNN Model
Overview
This project performs binary classification of MRI brain images to distinguish between normal and abnormal cases using deep learning. The dataset consists of 3D MRI brain images in DICOM format, and the task is approached using both a pretrained 3D CNN model and a custom 2D CNN.

The code is designed to allow flexibility in selecting either a pretrained 3D CNN (based on the 3D ResNet architecture) or a custom 2D CNN for binary classification. It also includes functionality for handling 3D medical images and augmentations, class balancing, and early stopping during training.

Features
Pretrained 3D CNN Model (ResNet-based): This model is used for 3D MRI volumes, fine-tuned for binary classification.
Custom 2D CNN Model: For cases where 2D slices from MRI volumes are used for training.
Augmentation Support: The dataset is augmented using various transformations (rotation, affine transforms, color jitter, and more) to improve generalization.
Class Imbalance Handling: Includes class weight calculation and weighted random sampling to address class imbalances in the dataset.
Early Stopping: Training stops early if the validation loss does not improve after a specified number of epochs, helping to avoid overfitting.
Model Saving: The best-performing model is saved for future predictions and inference.

Project Structure
.
├── data/
│   ├── X_train/        # Training data (3D MRI volumes in DICOM format)
│   ├── y_train/        # Training labels (0 or 1 for normal/abnormal)
│   ├── X_test/         # Test data (3D MRI volumes in DICOM format)
│   ├── y_test/         # Test labels (0 or 1 for normal/abnormal)
├── src/
│   ├── dataset.py      # Custom dataset class for handling 3D/2D MRI data
│   ├── model.py        # 3D and 2D CNN model definitions
│   ├── train.py        # Model training with early stopping and evaluation
│   ├── inference.py    # Script for loading the trained model and making predictions
│   └── utils.py        # Utility functions for data processing, augmentation, etc.
├── README.md           # Project overview and instructions
├── requirements.txt    # List of required Python packages
└── best_model.pth      # The best-performing model saved during training


Usage
1. Clone the repository
  git clone https://github.com/abolfazltorbat/mri-brain-classification.git
  cd mri-brain-classification

2. Install Dependencies
  pip install -r requirements.txt

3. Data Preparation
  Place your MRI brain images (DICOM format) in the data/ folder,
  organized into X_train, y_train, X_test, and y_test directories.
  Ensure the labels (y_train and y_test) are provided as binary
  values (0 for normal, 1 for abnormal).

4. Training the Model
   To train the model, choose whether to use the 3D or 2D CNN model by setting the use_3d_model
   flag in the train.py script:
  use_3d_model = True: Uses the pretrained 3D CNN model.
  use_3d_model = False: Uses the custom 2D CNN model.

Model Performance
The performance of the model is evaluated using accuracy, precision, recall, and F1 score, which are displayed after each validation step during training.

Future Improvements
Integrate advanced 3D image augmentations specific to medical images.
Experiment with different 3D CNN architectures (e.g., 3D DenseNet, 3D UNet).
Incorporate additional metadata (e.g., patient age, sex) into the model for better predictions.
License
This project is licensed under the MIT License.


