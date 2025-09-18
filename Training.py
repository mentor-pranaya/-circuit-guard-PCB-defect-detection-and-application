# Training Part

# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Data Loading and Preprocessing ---
# This part should be included in the training script as it prepares the data

# 1. Load the data from the CSV file
df = pd.read_csv('/content/drive/MyDrive/roi_labels.csv')

# 2. Create a new column with the full path to each image file including the subdirectory
df['image_path'] = '/content/drive/MyDrive/ROI_dataset/' + df['defect_type'] + '/' + df['roi_filename']

# Verify if the generated paths exist (optional for the final script but good for debugging)
df['file_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
print(f"Number of image files found: {df['file_exists'].sum()} out of {len(df)}")
if df['file_exists'].sum() < len(df):
    print("\nImage files not found for the following rows:")
    # In a script, you might want to handle this more robustly than just printing
    print(df[~df['file_exists']].head())


# 3. Map the string labels to numerical labels
unique_labels = df['defect_type'].unique()
label_mapping = {label: i for i, label in enumerate(unique_labels)}
df['numerical_label'] = df['defect_type'].map(label_mapping)

# Determine the number of classes
num_classes = len(label_mapping)


# 4. Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Define image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # Add any data augmentation techniques you want to use for training
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255) # No augmentation for validation/testing


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='defect_type',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    classes=list(label_mapping.keys()),
    shuffle=True
)

# The validation generator will be used for both validation during training and final testing
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='defect_type',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False, # Important: Do not shuffle validation/test data
    classes=list(label_mapping.keys())
)

print("Data generators created.")

# --- Model Definition ---
# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Create a Sequential model and add the base model
model = Sequential()
model.add(base_model)

# Add a Flatten layer and Dense layers for classification head
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print("Model defined.")

# --- Train the Classification Head ---
# Freeze the layers of the pre-trained base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

print("Model compiled for training classification head.")

# Train the compiled model
print("Training classification head...")
history = model.fit(
    train_generator,
    epochs=10, # Adjust epochs as needed
    validation_data=validation_generator
)

print("Classification head training completed.")


# --- Fine-tune the Model ---
# Unfreeze some layers in the base model for fine-tuning
unfreeze_layers = 4 # Example: unfreezing the last 4 layers of VGG16
for layer in base_model.layers[-unfreeze_layers:]:
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), # Use a lower learning rate
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

print(f"Top {unfreeze_layers} layers of the base model have been unfrozen for fine-tuning.")
print("Model recompiled with a lower learning rate for fine-tuning.")

# Continue training the entire model
fine_tune_epochs = 10 # Adjust fine-tuning epochs as needed
print("Fine-tuning the model...")
history_fine_tune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=validation_generator
)

print("Model fine-tuning completed.")

# Optionally save the trained model
# model.save('my_trained_model.h5')
