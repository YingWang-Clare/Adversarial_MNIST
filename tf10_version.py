"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    # mnist_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=20000,
    #     hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    index_of_2s = find_all_2s(eval_labels) # 1032
    x_batch = eval_data[index_of_2s[10:19]]
    # for i in eval_labels[index_of_2s]:
    #     if i == 2:
    #         continue
    #     else:
    #         print(i)
    # print('complete checking')
    plot_predictions(mnist_classifier, x_batch)

    # Pick a random 2 image from first 1000 images
    # Create adversarial image and with target label 6
    rand_index = np.random.randint(0, len(index_of_2s))
    image_norm = eval_data[index_of_2s[rand_index]]
    image_norm = np.reshape(image_norm, (1, 784))
    label_adv = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # one hot encoded, adversarial label 6
    # Plot adversarial images
    # Over each step, model certainty changes from 2 to 6
    # create_plot_adversarial_images(image_norm, label_adv, lr=0.2, n_steps=5)

def plot_predictions(mnist_classifier, image_list, output_probs=False, adversarial=False):
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": image_list},
        num_epochs=1,
        shuffle=False)
    pred_results = list(
        mnist_classifier.predict(input_fn=pred_input_fn, checkpoint_path='./tmp/mnist_convnet_model/model.ckpt-20200'))

    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)

    # Setup image grid
    cols = 3
    rows = int(math.ceil(image_list.shape[0] / cols))
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    # Get probs, images and populate grid
    for i in range(len(pred_results)):
        pred_list[i] = np.argmax(pred_results[i].get('probabilities'))  # for mnist index == classification
        pct_list[i] = pred_results[i].get('probabilities')[pred_list[i]] * 100

        image = image_list[i].reshape(28, 28)
        grid[i].imshow(image)

        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i],
                                  pct_list[i]))

        # Only use when plotting original, partial deriv and adversarial images
        # if (adversarial) & (i % 3 == 1):
        #     grid[i].set_title("Adversarial \nPartial Derivatives")

    plt.show()


def find_all_2s(labels):
    index_of_2s = [i for i in range(len(labels)) if labels[i] == 2]
    return index_of_2s

# def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=1, output_probs=False):
#     original_image = x_image
#     probs_per_step = []
#
#     # Calculate loss, derivative and create adversarial image
#     # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/gradient_computation
#     loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_conv)
#     deriv = tf.gradients(loss, x)
#     image_adv = tf.stop_gradient(x - tf.sign(deriv) * lr / n_steps)
#     image_adv = tf.clip_by_value(image_adv, 0, 1)  # prevents -ve values creating 'real' image
#
#     for _ in range(n_steps):
#         # Calculate derivative and adversarial image
#         dydx = sess.run(deriv, {x: x_image, keep_prob: 1.0})  # can't seem to access 'deriv' w/o running this
#         x_adv = sess.run(image_adv, {x: x_image, keep_prob: 1.0})
#
#         # Create darray of 3 images - orig, noise/delta, adversarial
#         x_image = np.reshape(x_adv, (1, 784))
#         img_adv_list = original_image
#         img_adv_list = np.append(img_adv_list, dydx[0], axis=0)
#         img_adv_list = np.append(img_adv_list, x_image, axis=0)
#
#         # Print/plot images and return probabilities
#         probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True)
#         probs_per_step.append(probs) if output_probs else None
#
#     return probs_per_step

if __name__ == "__main__":
    tf.app.run()
