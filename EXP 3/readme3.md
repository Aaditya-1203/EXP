Object 3
Program to implement a three-layer neural network using Tensor flow library (only, no keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches.
Description of the Model
Data Preprocessing:

The MNIST dataset is loaded and normalized (pixel values scaled between 0 and 1). Images are flattened from 28×28 to a 1D array of size 784. Labels are one-hot encoded for classification.

Model Architecture:

Input Layer: 784 neurons (one for each pixel in the image).
Hidden Layer 1: 128 neurons with sigmoid activation.
Hidden Layer 2: 64 neurons with sigmoid activation.
Output Layer: 10 neurons (one for each digit 0-9) with softmax activation.
Training Mechanism:

Uses Adam optimizer with a learning rate of 0.01. Cross-entropy loss function is used to measure classification error. Model is trained in mini-batches of size 64 over 10 epochs. Loss is calculated and updated at each step using gradient descent.

Performance Evaluation:
After training, the model is tested on the test dataset. Accuracy is computed by comparing predicted labels with actual labels. A loss curve is plotted to visualize how the model's loss decreases over epochs.

Visualization:
The model plots the loss curve to analyze training progress. Sample predictions are displayed to see how well the model recognizes handwritten digits

Code description :-
Preprocessing the MNIST Dataset
Normalization: Pixel values are scaled from 0-255 to 0-1 for faster convergence during training.
Flattening Images: The 28×28 grayscale images are converted into 1D arrays of 784 elements.
One-hot Encoding: Integer labels (0-9) are converted into one-hot vectors of size 10.
Defining the Neural Network
Input Layer: 784 neurons.
Hidden Layer 1: 128 neurons, initialized with random weights (W1) and biases (b1).
Hidden Layer 2: 64 neurons with (W2, b2).
Output Layer: 10 neurons for classification (W3, b3).
Matrix multiplication (tf.matmul()) applies weights and biases.
Sigmoid activation is used for hidden layers.
Softmax activation is used at the output layer for probability distribution over 10 classes.
Defining the Loss Function and Optimizer
Loss Function: Computes cross-entropy loss for classification.
Optimizer: Uses Adam optimizer with a learning rate of 0.01 for updating weights.
Defining the Training Step
Uses TensorFlow's GradientTape to compute gradients for backpropagation.
The weights and biases are updated using the computed gradients.
Evaluating the Model on Test Datasets
Innference is performed on the test dataset and accuracy is calculated.

Training Loop
Training is carried out for 10 epochs.
Batch size is set to 64.
Stores loss values for visualization.
Training is carried out in mini-batches:
a. Loss is computed and weights are updated for each batch .
b. The progress bar is updated with current loss.
c. Loss per epoch is stored for visualisation.
Evaluating the Model After Training
The final test accuracy after training is calculated.
Visualisation
The loss curve is plotted to track training performance.
The first 5 images are selected from the test set.
Each image is displayed along with the model's predicted class and actual class.
My Comments
The model may take a lot of time to train for e.g. A 6gb vram graphics card (Nvidia RTX 4050 Laptop GPU) takes around 2 minutes for 10 epochs and batch size 64. In order to improve the training time the hardware should be equipped with the robust specifications.
Instead of using the sigmoid function as activation function, rectified linear unit (ReLu) function can be used which is comparatively simple and hence improves training time.
More layer can be added to learn complicated patterns and improve accuracy.
