import os  # Import the tool used to check whether the model file exists.
import tkinter as tk  # Import the Tkinter user interface toolkit.
from tkinter import filedialog, messagebox  # Import file selection and error dialogs.

import cv2  # Import the tool used to read and prepare images.
import numpy as np  # Import the tool used to work with numeric arrays.
import tensorflow as tf  # Import the main deep learning library.
from PIL import Image, ImageTk  # Import tools used to display the selected image.
from keras import datasets  # Import the built-in Keras datasets.
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D  # Import the model layers.
from keras.models import Sequential  # Import the sequential model type.

# Define the path where the trained model is stored.
MODEL_PATH = 'model.h5'

# Define the class names included in CIFAR-10.
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Create the convolutional neural network structure.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

# Configure the model with an optimizer, loss function, and accuracy metric.
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Load the model if it exists; otherwise, train and save a new model.
if os.path.exists(MODEL_PATH):
    # Load the weights from the existing model.
    model.load_weights(MODEL_PATH)
    print('The existing model was loaded; training was skipped.')
else:
    # Load the CIFAR-10 dataset only when training is necessary.
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values from the range 0-255 to the range 0-1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Train the network when no saved model exists.
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels)
    )

    # Save the trained model for future runs.
    model.save(MODEL_PATH)
    print('Model training finished and the model was saved.')


def predict_image(file_path):
    # Read the selected image from the provided file path.
    image = cv2.imread(file_path)

    # Stop and show an error if the selected file is not a readable image.
    if image is None:
        messagebox.showerror('Invalid image', 'The selected file could not be read.')
        return None

    # Convert the image from BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the model input size.
    image = cv2.resize(image, (32, 32))

    # Normalize pixel values to match the training data.
    image = image.astype('float32') / 255.0

    # Add a batch dimension and ask the model for a prediction.
    prediction = model.predict(np.array([image]), verbose=0)

    # Return the name of the class with the highest prediction score.
    return class_names[int(np.argmax(prediction[0]))]


def choose_image():
    # Open a file picker so the user can select an image.
    file_path = filedialog.askopenfilename(
        title='Select an image',
        filetypes=[
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.webp'),
            ('All files', '*.*')
        ]
    )

    # Do nothing when the user closes the file picker without selecting a file.
    if not file_path:
        return

    # Predict the class of the selected image.
    predicted_class = predict_image(file_path)

    # Stop if the image could not be read successfully.
    if predicted_class is None:
        return

    # Load the original image and scale it for the application window.
    selected_image = Image.open(file_path)
    selected_image.thumbnail((500, 350))
    image_preview = ImageTk.PhotoImage(selected_image)

    # Keep a reference to the image so Tkinter does not remove it from memory.
    image_label.image = image_preview
    image_label.configure(image=image_preview)

    # Show the selected file and the prediction result in the window.
    file_label.configure(text=os.path.basename(file_path))
    result_label.configure(text=f'Prediction: {predicted_class}')


# Create the main Tkinter window.
window = tk.Tk()
window.title('Image Classifier')
window.geometry('600x520')
window.resizable(False, False)

# Create the title shown at the top of the window.
title_label = tk.Label(window, text='CIFAR-10 Image Classifier', font=('Arial', 18, 'bold'))
title_label.pack(pady=15)

# Create the button that opens the image file picker.
select_button = tk.Button(window, text='Select Image', command=choose_image, width=20)
select_button.pack(pady=10)

# Create the label used to display the selected image.
image_label = tk.Label(window, text='No image selected', width=60, height=18)
image_label.pack(pady=10)

# Create the label used to display the selected file name.
file_label = tk.Label(window, text='')
file_label.pack(pady=5)

# Create the label used to display the prediction result.
result_label = tk.Label(window, text='Prediction: -', font=('Arial', 14, 'bold'))
result_label.pack(pady=10)

# Start the Tkinter event loop and wait for user interaction.
window.mainloop()