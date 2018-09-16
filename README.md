# Adversarial_MNIST
This project aims to create adversarial images to fool a MNIST classifier in TensorFlow.

## Requirements
To run the `main.py`, the following packages needs to be installed.
* python 2.7
* tensorflow 1.10.1
* numpy 1.14.5
* matplotlib 2.2.2

## File Description
* `Adversarial_MNIST.ipynb` is the Jupyter Notebook with the code divided into several cells. A good illustration of each step and function of the code.
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
