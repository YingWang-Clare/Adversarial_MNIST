# Adversarial MNIST
This project aims to create adversarial images to fool a MNIST classifier in TensorFlow.

## Requirements
To run the `main.py`, the following packages need to be installed.
* python 2.7
* tensorflow 1.10.1
* numpy 1.14.5
* matplotlib 2.2.2

## File Description
* `Adversarial_MNIST.ipynb` is the Jupyter Notebook with the code divided into several cells. A good illustration of each step and function.
* `main.py` is the python code for creating the adversarial images.
* `MNIST-data` contains the MNIST dataset.
* `tmp` folder contains the checkpoints of the MNIST classifier that I have already trained.

## Usage
After letting your environment satisfy the requirements, execute command `python main.py` in the terminal.

## Important Notes
In `Adversarial_MNIST.ipynb`, the version of tensorflow being used is 1.3.0. Thus, it will throw the error:

`AttributeError: 'Estimator' object has no attribute 'get_variable_value'`

when executing the function `adversarial_image()` in the last cell.

However, everything works well in `main.py` since the lastest version of **tensorflow 1.10.1** is used.

If you want to run the code and check the final resulting adversarial images, please do run the python file `main.py` with tensorflow 1.10.1.

## Limitations
In the project, I aimed to create adversarial images of 2 to fool the existing CNN model and make it predict all 2 as 6 with high certainty. But after exploring and attempting so many methods, finally, with each step, the certainty of the model recognizing image 2 is becoming lower and lower, as you can check in `result_images` folder. However, the model still cannot recognize the adversarial images as 6 but some other random labels (for example, sometimes 8 sometiems 3). Although I took the rationale of doing the gradient descent as training the neural network, but this time takes the derivatives of loss regarding input image. So, I'm currently still trying to findout the reason causing this phenomenon.
