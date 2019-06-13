# Customisable_Neural_Networks
Program that allows you to build your own Dense Neural Network with with dynamic visualisation.

## How it works

The program runs mainly on NumPy and the network is built from scratch, using the user's input as hyperparamenters for the Network. Allowing the user to choose his own dataset, amount of layers, and nodes per layer. The network is visualised once the user has given the program all of the necessary inputs. Tensorflow and Scikit learn are only imported to conveniently access the Make Moons & MNIST dataset.

This will look something like this:

<img align="center" width="400" src="https://github.com/LeanderNicolai/Customisable_Neural_Networks/blob/master/NN_Vis.png">     <img align="center" width="400"  src="https://github.com/LeanderNicolai/ArtificialNeuralNetworks/blob/master/KDP.gif">

So far the layers in the network are only dense layers, using a Sigmoid as an activation function. Convolutional Layers and a ReLu activation function are to be implemented in the future.

## How to use it
1. **Clone** this repository
2. Install the requirements by running the following command in your shell while in this repository:

```pip install -r requirements.txt```

3. Run **clean_dynamic_nn.py**
4. Choose the dataset you would like to train the network on. It is possible to choose a standard dataset, like the Make_Moons dataset, or the MNIST dataset.
5. If you want to import your own dataset, it is important to have it in the repository folder as a .CSV with the labels as the final column
6. Choose if you would like to have see have a dynamic Kernel Density Plot of the training cycle. If you have more than 2 features, the model will choose two random features for the Plot. The command line will show the exact name of the Plot once the training is complete.
7. Choose for how many epochs you would like to train the model
8. **Now for the fun part!** Here you are able to compose your own Neural Network!
9. Enter how many **Layers** you would like to have in your network
10. Enter how many **Nodes** you want for each layer
11. Take a look at the network you've created. It's beautiful!
12. Press enter to start training!
13. Once you are done training you can make predictions with the weights and biases generated from the training cycle.
