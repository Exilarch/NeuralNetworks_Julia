# NeuralNetworks_Julia

Do this in a Jupyter notebook so the graphs will display properly

The main function, train_nn, has the following signature:

```
train_nn(layers_dimensions, activation_functions, X , Y , learning_rate , n_iter, lambda)
```

These inputs are:

* layers_dimensions - the number of layers and neurons in each layer
* activation_functions - the activation function to use for each (non-input) layer
* X - input features, as x<sub>n</sub> x m matrix
* Y - output values, as 1 x m matrix
* learning_rate - $\alpha$
* n_iter - number of iterations to train neural network
* lambda - regularization hyperparameter (ridge regression)
