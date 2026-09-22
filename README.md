# CIFAR-10 Image Classifier

This project is an image classification application built using a convolutional neural network (CNN) and the CIFAR-10 dataset. The app lets the user select an image, then predicts its class and displays the result in a Tkinter window.

## Features

- Select an image from the computer using the `Select Image` button
- Display the selected image in the application window
- Detect the image class and show the prediction result
- Load a saved model to avoid retraining every time
- Train the model automatically if `model.h5` does not exist
- Save the trained model after the first training session

## Recognizable Classes

The CIFAR-10 model can detect these classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install tensorflow keras opencv-python numpy pillow
```

## Run the App

```bash
python index.py
```

After running the application:

1. Click the `Select Image` button.
2. Choose an image from your computer.
3. The image is displayed in the app window.
4. The prediction result appears in the `Prediction` section.

## Model Management

- If `model.h5` already exists, the model is loaded and training is skipped.
- If `model.h5` does not exist, the CIFAR-10 dataset is downloaded and the model is trained.
- After training, the model is saved to `model.h5` so it does not need to be retrained on the next run.

## Important Note

This model was trained on CIFAR-10 images of size 32x32 pixels. Therefore, accuracy may be limited on real-world or more complex images. Before prediction, the selected image is resized to 32x32.
