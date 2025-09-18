# Testing Part

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # To load a saved model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Data Loading and Preprocessing (required for setting up test_generator) ---
# This part duplicates some of the loading/preprocessing for creating the test generator
# In a real scenario, you would load the dataframes saved during training or
# ensure this preprocessing is consistent.

# 1. Load the data from the CSV file
df = pd.read_csv('/content/drive/MyDrive/roi_labels.csv')

# 2. Create a new column with the full path to each image file including the subdirectory
df['image_path'] = '/content/drive/MyDrive/ROI_dataset/' + df['defect_type'] + '/' + df['roi_filename']

# Verify if the generated paths exist (optional for the final script but good for debugging)
df['file_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
print(f"Number of image files found: {df['file_exists'].sum()} out of {len(df)}")
if df['file_exists'].sum() < len(df):
    print("\nImage files not found for the following rows:")
    print(df[~df['file_exists']].head())

# 3. Map the string labels to numerical labels
unique_labels = df['defect_type'].unique()
label_mapping = {label: i for i, label in enumerate(unique_labels)}
df['numerical_label'] = df['defect_type'].map(label_mapping)

# 4. Split the DataFrame into training and testing sets (needed to get the test_df)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define image dimensions (should match the model definition)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Create a data generator for testing (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='defect_type',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False, # Important: Do not shuffle test data
    classes=list(label_mapping.keys())
)

print("Test data generator created.")

# --- Model Loading ---
# If you saved your model during training, load it here
# model = load_model('my_trained_model.h5')

# If running the testing part immediately after training in the same script,
# the 'model' variable will already be available from the training part.
# Make sure the model variable exists if running this code standalone.


# --- Model Evaluation ---
print("Evaluating the model on the test set...")
evaluation_results = model.evaluate(test_generator)

# Print the evaluation results
print("\nTest Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])
