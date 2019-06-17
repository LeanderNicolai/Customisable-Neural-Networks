import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import numpy as np
from math import e
from math import log as log
import os
import imageio
import datetime
import seaborn as sns
from sklearn.metrics import accuracy_score
import sklearn
import start_nvs as NVStart
import weighted as WV
from keras.datasets import mnist


def choose_dataset():
    choice = int(input('''Which dataset would you like to train your network on?

    For Make Moons:     Press 1
    For MNIST:          Press 2
    To import your own Dataset: Press 3

    '''))
    if choice == 1:
        X, ytrue = make_moons(n_samples=200, noise=0.2, random_state=42)
        Xtest, ytest = make_moons(n_samples=200, noise=0.2, random_state=42)
        make_kde = True
        KDP_INPUT = input('Do you want a KDP Plot?  Y / N  ')
        KDP_INPUT = KDP_INPUT.upper()
        if 'Y' in KDP_INPUT:
            KDP_FLAG = True
            print("KDP Flag set to True")
        else:
            KDP_FLAG = False
    else:
        (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
        X = xtrain.reshape(60000, 784)
        X = X[:1000, :]
        ytrue = ytrain[:1000]
        Xtest = xtest.reshape(10000, 784)
        Xtest = Xtest[1000:1500, :]
        ytest = ytrain[1000:1500]
        KDP_FLAG = False
    return X, ytrue, Xtest, ytest, choice, KDP_FLAG


def KDP(ypred, i, X, desc_loss, acc):
    idx_1 = np.where(ypred == 1)
    idx_0 = np.where(ypred == 0)
    zero_y_one = X[:, 0][idx_1]
    zero_y_two = X[:, 0][idx_0]
    one_y_one = X[:, 1][idx_1]
    one_y_two = X[:, 1][idx_0]
    print("zero_y_one shape is ", zero_y_one.shape)
    print("zero_y_two shape is ", zero_y_two.shape)
    plt.figure()
    sns.kdeplot(zero_y_one, one_y_one, shade=False, kde=True)
    sns.kdeplot(zero_y_two, one_y_two, shade=False, kde=True)
    plot = plt.scatter(X[:, 0], X[:, 1], c=ypred)
    plt.title(f'Epoch: {i} Loss: {desc_loss:.2f} Accuracy:  {acc}')
    filename = 'lifeexp_{}.png'.format(i)
    return filename


def shape_creation(X, y):
    input_layer_nc = X.shape[1]
    output_layer_nc = len(np.unique(ytrue))
    testing = input('Press Y for custom network and N for standard:    ')
    testing = testing.upper()
    print(testing)
    if testing == 'Y':
        layer_amount = int(input('How many hidden layers do you want your network to have?    '))
        layer_nc = []
        for i in range(layer_amount):
            node_count = int(input(f'How many nodes do you want node{i + 1} to have?    '))
            layer_nc.append(node_count)
        final_nc = int(input('How many nodes should the output layer have?    '))
        layer_nc.append(final_nc)
    else:
        layer_nc = [4, 2, 5, 3]
        final_nc = int(input('How many nodes should the output layer have?    '))
        layer_nc.append(final_nc)
    i = 0
    weight_shapes = []

    for layer in layer_nc:
        if i == 0:
            weight_shapes.append((layer, input_layer_nc))
            i = i + 1
            continue
        hw_shape = (layer_nc[i], layer_nc[i-1])
        weight_shapes.append(hw_shape)
        i = i + 1
    if max(layer_nc) < 100:
        network = NVStart.DrawNN(layer_nc)
        network.draw()
        print("Drawing the Network Now")
    else:
        print("Network is too large to visualize")
    return weight_shapes, layer_nc


def initialize_weights(weight_shapes):
    bias_shapes = [np.random.normal(size=((x[0], 1))) for x in weight_shapes]
    all_weights = [np.random.normal(size=(x)) for x in weight_shapes]
    return all_weights, bias_shapes


def sigmoid(x):
    return 1 / (1 + e**(-x))


def sigmoid_prime(y):
    return sigmoid(y) * (1-sigmoid(y))


def loss(ytrue, ypred):
    return -((ytrue*np.log(ypred)) + ((1 - ytrue)*np.log(1 - ypred)))


def feed_forward(X_T, all_weights, bias_shapes):
    outputs = []
    outputs.append(X_T)
    for i in range(len(all_weights)):
        latest_output = sigmoid(np.dot(all_weights[i], outputs[-1]))
        dot_out = np.dot(all_weights[i], outputs[-1])
        latest_output = sigmoid(dot_out + bias_shapes[i])
        outputs.append(latest_output)
    return outputs


def back_prop(outputs, ytrue, all_weights, bias_shapes):
    i = 0
    upd_w = []
    new_biases = []
    out_rev = outputs
    weights_rev = all_weights[:]
    weights_rev.reverse()
    out_rev.reverse()
    bias_rev = bias_shapes[:]
    bias_rev.reverse()
    gradients = []
    LR = 0.05
    for i in range(len(out_rev) - 1):
        if i == 0:
            error = (out_rev[0] - ytrue) * loss(ytrue, out_rev[0])
            grad_y = out_rev[0] * error
            gradients.append(grad_y)
            weights_delta = np.dot(-gradients[-1], out_rev[1].T) * LR
            bias_delta = np.sum(-gradients[-1] * bias_rev[0]) * LR
            updated_bias = bias_delta + bias_rev[0]
            new_biases.append(updated_bias)
            updated_weights = weights_rev[0] + weights_delta
            upd_w.append(updated_weights)

        else:
            right_h = np.dot(weights_rev[i-1].T, gradients[-1])
            grad_ho = out_rev[i] * right_h
            gradients.append(grad_ho)
            weights_delta = np.dot(-gradients[-1], out_rev[i+1].T) * LR
            bias_delta = np.sum(-gradients[-1] * bias_rev[i]) * LR
            updated_bias = bias_delta + bias_rev[i]
            new_biases.append(updated_bias)
            updated_weights = weights_rev[i] + weights_delta
            print('changed weights by: ', weights_delta.sum())
            upd_w.append(updated_weights)
        i += 1
    return upd_w, bias_shapes, new_biases


def create_network(X, y):
    X_T = X.T
    y = ytrue
    weight_shapes, layer_nc = shape_creation(X, y)
    all_weights_1, bias_shapes = initialize_weights(weight_shapes)
    return all_weights_1, bias_shapes, X_T, ytrue, layer_nc


def train_network(epochs, X_T, ytrue, all_weights_1, bias_shapes, choice, KDP_FLAG, layer_nc):
    images = []
    dynamic = []
    cwd = os.getcwd()
    for i in range(epochs):
        outputs = feed_forward(X_T, all_weights_1, bias_shapes)
        outputs_cp = outputs
        all_weights_1, bias_shapes, new_biases = back_prop(
            outputs, ytrue, all_weights_1, bias_shapes)
        bias_shapes[0] = new_biases[0]
        all_weights_1.reverse()
        to_shape = outputs[0].T.shape[0]
        ypred = outputs[0].T.reshape(to_shape)
        ypred_int = ypred.round()
        desc_loss = loss(ytrue, ypred).sum()
        print(ypred_int)
        acc = sklearn.metrics.accuracy_score(ytrue, ypred_int)
        print('loss is:  ', desc_loss)
        print('accuracy is:  ', acc)
        X = X_T.T
        print(X.shape)
        print(type(ypred_int))
        print(ypred_int)
        if KDP_FLAG:
            filename = KDP(ypred_int, i, X, desc_loss, acc)
            file_loc = cwd + '/pngs/' + filename
            plt.savefig(filename)
            images.append(imageio.imread(filename))
        weights = all_weights_1
        w2 = weights[1:]
        network = WV.DrawNN(layer_nc, w2, new_biases)
        fig = network.draw()
        fname = f"dynamic_plot_{i}.png"
        plt.savefig(fname)
        dynamic.append(imageio.imread(fname))
    if KDP_FLAG:
        time = str(datetime.datetime.now())
        file_ext = time[-2:]
        imageio.mimsave(f'output{file_ext}.gif', images, fps=15)
        print(f"Kernel Density Plot saved as: output{file_ext}.gif")
    imageio.mimsave(f'dynamic_train_13.gif', dynamic, fps=15)
    for file in os.listdir(cwd):
        if file.endswith(".png"):
            print("removed", file)
            os.remove(file)

    return all_weights_1, bias_shapes


def create_train(X, ytrue, choice, KDP_FLAG):
    epochs = int(input('How many iterations would you like to run?    '))
    X_T = X.T
    all_weights_1, bias_shapes, X_T, ytrue, layer_nc = create_network(X, ytrue)
    weights, biases = train_network(epochs, X_T, ytrue, all_weights_1,
                                    bias_shapes, choice, KDP_FLAG, layer_nc)

    return weights, biases, X_T


def predict(X_T, all_weights, bias_shapes):
    outputs = feed_forward(X_T, all_weights, bias_shapes)
    outputs.reverse()
    to_shape = outputs[0].T.shape[0]
    ypred = outputs[0].T.reshape(to_shape)
    ypred_int = ypred.round()
    return outputs, ypred_int


X, ytrue, Xtest, ytest, choice, KDP_FLAG = choose_dataset()

weights, biases, X_T = create_train(X, ytrue, choice, KDP_FLAG)

outputs, ypred_int = predict(X_T, weights, biases)
