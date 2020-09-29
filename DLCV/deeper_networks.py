#!/usr/bin/env python3

# AlexNet (pseudo mini one)
# VGG
# NIN: https://arxiv.org/abs/1312.4400
# Inception
# ResNet
# ?GoogLeNet?
# ?Xception?
# ?Attention? Spatial Transformer Network: https://arxiv.org/abs/1506.02025
# ?EfficientNet?

# Compare: number of trainable parameters; max. train/test accuracy;
#    plot of parameters vs. accuracies
# confusion matrices to show weakspots in classifiers, NOT JUST AVG ACCURACY
# Data: MNIST, CIFAR-10, Tiny-ImageNet

# Papers from module: http://www.eecs.qmul.ac.uk/~sgg/_ECS795P_/papers/

#TODO check that we are running TF >= 2??
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model

import json
import matplotlib.pyplot as plt
# %matplotlib inline

import argparse
import itertools
import numpy as np
import os.path

class MNIST:
    def __init__(self):
        self.name = "MNIST"

        from tensorflow.keras.datasets import mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        print("#train:", self.n_train, "#test:", self.n_test)

        self.instance_shape = self.x_train.shape[1:]
        if len(self.instance_shape) == 2:
            """
            The MNIST data comes in (28, 28) per instance rather than then (28, 28, 1) that
            will be required by a Conv2D layer so here we reshape appropriately to explicitly
            define a third channel of depth 1
            """
            self.instance_shape = (*self.instance_shape, 1)

            self.x_train = self.x_train.reshape((self.n_train, *self.instance_shape))
            self.x_test = self.x_test.reshape((self.n_test, *self.instance_shape))

        print("instance-shape:", self.instance_shape)

        # normalise the pixel intensities into 0..1
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        self.n_classes = self.y_train.shape[1]
        self.class_labels = [ str(x) for x in range(self.n_classes) ]
        print("#classes", self.n_classes, ":", self.class_labels)

    def show_instance(self):
        k = 666
        i = self.x_train[k, :, :, 0]
        plt.imshow(i, cmap = "binary")
        print(self.y_train[k])
        plt.show()


class CIFAR10:
    def __init__(self):
        self.name = "CIFAR10"

        from tensorflow.keras.datasets import cifar10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        print("#train:", self.n_train, "#test:", self.n_test)

        self.instance_shape = self.x_train.shape[1:]
        print("instance-shape:", self.instance_shape)

        # normalise the pixel intensities into 0..1 --- significantly improves convergence
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        self.n_classes = self.y_train.shape[1]

        # the Keras way of accessing the CIFAR10 data doesn't include the text labels
        # so we're just going to hard-code them here, from: https://www.cs.toronto.edu/~kriz/cifar.html
        self.class_labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" ]

        print("#classes", self.n_classes, ":", self.class_labels)

    def show_instance(self):
        k = 666
        i = self.x_train[k, :, :, :]

        plt.imshow(i)

        j = np.argmax(self.y_train[k])
        plt.title(self.class_labels[j])

        plt.show()


class CONV:
    def __init__(self, dataset, scaling, dropout = False):
        self.dataset = dataset
        self.scaling = scaling
        self.dropout = dropout

        if self.dropout:
            name = "DROP"
        else:
            name = "CONV"

        self.name = "{}.{}x{}".format(name, dataset.name, dataset.instance_shape[0] * scaling)

    def build_model(self):
        # a simple network in the style of VGG-A but with many fewer layers and
        # convolutional filters. Still manages >95% training accuracy in ~5 epochs
        # on MNIST
        # https://arxiv.org/abs/1409.1556

        #TODO compare to model from §5.1 of Chollet book

        model = keras.Sequential()

        if self.scaling is not None and self.scaling > 1:
            model.add(UpSampling2D(size = self.scaling, input_shape = self.dataset.instance_shape, interpolation = "bilinear"))
        else:
            model.add(Input(shape = self.dataset.instance_shape))

        model.add(Conv2D(filters = 32, kernel_size = 3, activation = "relu"))
        model.add(MaxPooling2D(pool_size = 2))

        model.add(Conv2D(filters = 64, kernel_size = 3, activation = "relu"))
        model.add(MaxPooling2D(pool_size = 2))

        model.add(Conv2D(filters = 64, kernel_size = 3, activation = "relu"))

        model.add(Flatten())

        model.add(Dense(units = 64, activation = 'relu'))

        if self.dropout:
            model.add(Dropout(0.5))

        model.add(Dense(units = self.dataset.n_classes, activation = 'softmax'))

        model.summary()

        return model


class NIN:
    """
    Network-in-Network
    https://arxiv.org/abs/1312.4400

    From original paper:
        consist of three stacked mlpconv layers, and
        the mlpconv layers in all the experiments are followed by a spatial max pooling layer which
        downsamples the input image by a factor of two. As a regularizer, dropout is applied on the outputs of all
        but the last mlpconv layers.

        Another regularizer applied is weight decay as used by Krizhevsky et al. [4].
        n.b

        Within each mlpconv layer, there is a three-layer perceptron. The number of layers in both NIN and the
        micro networks is flexible and can be tuned for specific tasks.

    Some example code and discussion here: https://stats.stackexchange.com/questions/273486/network-in-network-in-keras-implementation
    However that mixes Max/Avg pooling between mlpconv layers; also doesn't appear to use regularisation

    "detailed settings of the parameters are provided in the supplementary materials."
    TODO find these... otherwise we have to work from the stackexchange article ...

    """
    def __init__(self, dataset, scaling):
        self.dataset = dataset
        self.scaling = scaling

        self.name = "NIN.{}x{}".format(dataset.name, dataset.instance_shape[0] * scaling)

    def build_model(self):
        model = keras.Sequential()

        #TODO review layer sizes etc. if possible to find original paper supplementary material

        #TODO What was the input shape used by the authros?
        if self.scaling is not None and self.scaling > 1:
            model.add(UpSampling2D(size = self.scaling, input_shape = self.dataset.instance_shape, interpolation = "bilinear"))
        else:
            model.add(Input(shape = self.dataset.instance_shape))

        # With 64 filters in all mlpconv blocks we can do well at MNIST but only ~64% on CIFAR10 test
        # data.
        # 128 gets us only to 80% for CIFAR10
        n_filters = 64

        # First mlpconv block
        # The 3-layer sliding-MLP component implemented using 1x1 convolutional layers
        model.add(Conv2D(filters = n_filters, kernel_size = 5, activation = "relu"))
        model.add(Conv2D(filters = n_filters, kernel_size = 1, activation = "relu"))
        model.add(Conv2D(filters = n_filters, kernel_size = 1, activation = "relu"))

        # between NIN blocks we have max-pooling
        model.add(MaxPooling2D(pool_size = 2))
        # n.b. this seems fundamental to success with MNIST! Without we get ~80% on train and 78 on test
        #      with dropout we get 99%+ ...
        #      The example had a 70% dropout rate...
        model.add(Dropout(rate = 0.7))

        # Second mlpconv block, n.b. if kernel-size is 5 then we don't learn anything ... TODO why?!?
        n_filters = 192
        model.add(Conv2D(filters = n_filters, kernel_size = 3, activation = "relu"))
        model.add(Conv2D(filters = n_filters, kernel_size = 1, activation = "relu"))
        model.add(Conv2D(filters = n_filters, kernel_size = 1, activation = "relu"))
        model.add(MaxPooling2D(pool_size = 2))
        model.add(Dropout(rate = 0.7))

        # Final mlpconv block
        n_filters = 256
        model.add(Conv2D(filters = n_filters, kernel_size = 3, activation = "relu"))
        model.add(Conv2D(filters = n_filters, kernel_size = 1, activation = "relu"))
        # n.b. size of final layer in last mlpconv must be n_classes
        model.add(Conv2D(filters = self.dataset.n_classes, kernel_size = 1, activation = "relu"))

        model.add(GlobalAveragePooling2D())
        model.add(Activation("softmax"))

        model.summary()

        return model


class VGG:
    """
    https://neurohive.io/en/popular-networks/vgg16/
    """
    def __init__(self, dataset, scaling):
        self.dataset = dataset
        self.scaling = scaling
        self.variant = "A"

        self.name = "VGG-{}.{}x{}".format(self.variant, dataset.name, dataset.instance_shape[0] * scaling)

    def build_model(self):
        model = keras.Sequential()

        # our datasets are spatially small but the VGG architecture is designed to work
        # with 224x224 images. So we upscale if there's a nice integer scaling we can do
        # (thankfully we can with MNIST and CIFAR10).
        # This is almost definitely massively inefficient and we should instead revise the
        # network architecture to have fewer layers of pooling while still being able to
        # capture the important features
        #
        # this link uses the raw 28x28 for MNIST: https://medium.com/@amir_hf8/implementing-vgg13-for-mnist-dataset-in-tensorflow-abc1460e2b93
        # https://github.com/amirhfarzaneh/vgg13-tensorlfow
        # but using the raw 32x32 for CIFAR yields a flatlining loss

        if self.scaling is not None and self.scaling > 1:
            model.add(UpSampling2D(size = self.scaling, input_shape = self.dataset.instance_shape, interpolation = "bilinear"))
        else:
            model.add(Input(shape = self.dataset.instance_shape))

        """
        the spatial padding of conv. layer input is such that the spatial resolution is preserved
        after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers. Spatial pooling is carried out by
        five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed
        by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
        """

        model.add(Conv2D(filters = 64, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = "same"))

        model.add(Conv2D(filters = 128, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = "same"))

        model.add(Conv2D(filters = 256, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(Conv2D(filters = 256, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = "same"))

        model.add(Conv2D(filters = 512, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(Conv2D(filters = 512, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = "same"))

        model.add(Conv2D(filters = 512, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(Conv2D(filters = 512, kernel_size = 3, activation = "relu", padding = "same"))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = "same"))

        model.add(Flatten())
        # dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5)
        # pre-built code in Keras doesn't include the Dropout for some reason ...

        # It didn't seem clear whether dropout appeared before or after each hidden dense layer
        # but this PyTorch version of VGG suggests after each hidden layer, so we will follow that
        # https: // github.com / pytorch / vision / blob / master / torchvision / models / vgg.py
        # add L2 regularisation to the fully-connected layers of .0005
        weight_decay = 0.0005

        model.add(Dense(units = 4096, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay)))
        model.add(Dropout(rate = 0.5))
        model.add(Dense(units = 4096, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay)))
        model.add(Dropout(rate = 0.5))
        model.add(Dense(units = self.dataset.n_classes, activation = 'softmax'))

        model.summary()

        return model


class RES:
    def __init__(self, dataset, scaling, depth = 20):
        self.dataset = dataset
        self.scaling = scaling
        self.depth = depth

        self.name = "RES{}.{}x{}".format(self.depth, dataset.name, dataset.instance_shape[0] * scaling)

    def build_model(self):
        # based on https://keras.io/examples/cifar10_resnet/

        # Can't use a sequential model here as the skip/identity connections need
        # Keras functional model
        # model = keras.Sequential()

        if self.scaling is not None and self.scaling > 1:
            x = UpSampling2D(size = self.scaling, input_shape = self.dataset.instance_shape, interpolation = "bilinear")
        else:
            x = Input(shape = self.dataset.instance_shape)

        model_inputs = x

        # add Conv -> BatchNorm [-> ReLU] layers with suitable regulariser etc.
        def layer(inputs, n_filters, kernel_size = 3, strides = 1, activation = "relu", batchnorm = True):
            conv = Conv2D(n_filters,
                          kernel_size = kernel_size,
                          strides = strides,
                          padding = 'same',
                          kernel_initializer = 'he_normal',
                          kernel_regularizer = keras.regularizers.l2(1e-4))

            y = conv(inputs)
            if batchnorm:
                y = BatchNormalization()(y)

            if activation is not None:
                y = Activation(activation)(y)

            return y

        def block(inputs, n_filters, downscaling = False):
            stride_length = 2 if downscaling else 1

            y = layer(inputs, n_filters, strides = stride_length)
            y = layer(y, n_filters, activation = None)  # no activation as we add identity before ReLU

            if downscaling:
                x = layer(inputs, n_filters, kernel_size = 1, strides = stride_length, activation = None, batchnorm = False)
            else:
                x = inputs

            y = keras.layers.add([ x, y ])  # add together the identity and the post-conv values
            return Activation("relu")(y)

        n_filters = 16
        x = layer(x, n_filters)

        # the ResNet paper shows a 34-layer residual network with 16 residual blocks
        # each of which contains 2 Conv2D layers and the network then adds an initial
        # and final layer.
        # So if we're having 3 levels of n_filters then for a "ResNet20" we need
        # 6 Convolutional layers == 3 residual-blocks for each chunk/stack
        # i.e. (depth - 2) / (3-chunks * 2-layers-per-block)
        # The 1x1 convolutional "identity" layers aren't included in this count
        n_blocks = int((self.depth - 2) / 6)

        for i_stack in range(3):
            for i_block in range(n_blocks):
                x = block(x, n_filters, i_stack > 0 and i_block == 0)

            n_filters *= 2

        # The Keras example uses AveragePooling2D(pool_size=8) but the paper
        # specifically states global average pooling so we'll run with that
        y = GlobalAveragePooling2D()(x)
        y = Dense(units = self.dataset.n_classes, activation = 'softmax')(y)

        model = keras.models.Model(model_inputs, y)
        model.summary()

        return model


def main(argv):
    parser = argparse.ArgumentParser()
    #parser.add_argument("--session", required = True)
    parser.add_argument("--epochs", default = 1, type = int, help = "Number of epochs to run")
    parser.add_argument("--restart", action = 'store_true', default = False, help = "Train ignoring any existing saved model")
    parser.add_argument("--resume", action = 'store_true', default = False, help = "Train an existing saved model some more")

    parser.add_argument("--all-data", action = 'store_true', default = False, help = "Run all datasets")
    parser.add_argument("--all-arch", action = 'store_true', default = False, help = "Run all architectures")

    # choice of which optimiser to use, n.b. will only take effect on newly started
    # or re-started configurations, not on resumed
    parser.add_argument("--sgd", action = 'store_true', default = False, help = "Use SGD optimiser, default: Adam, (only for new or restarting models)")

    parser.add_argument("--mnist", action = 'store_true', default = False, help = "Use MNIST dataset")
    parser.add_argument("--cifar10", action = 'store_true', default = False, help = "Use CIFAR-10 dataset")

    # optional upscaling of the source images by an integer multiple
    # only used on a per-architecture basis, especially for exploring
    # VGG's quality depending on the scaling
    parser.add_argument("--scaling", default = 1, type = int, help = "Upscale input images (only for new or restarting models)")

    # two baseline models: simple convnet and same but with dropout
    parser.add_argument("--conv", action = 'store_true', default = False, help = "Run baseline simple convnet")
    parser.add_argument("--dropout", action = 'store_true', default = False, help = "Run baseline simple convnet with dropout")

    # VGG
    parser.add_argument("--vgg", action = 'store_true', default = False, help = "Run VGG-A (11 layers) arch")

    # Network-in-network
    parser.add_argument("--nin", action = 'store_true', default = False, help = "Run Network-in-Network arch")

    # ResNet
    parser.add_argument("--res20", action = 'store_true', default = False, help = "Run 20-layer ResNet")
    parser.add_argument("--res32", action = 'store_true', default = False, help = "Run 32-layer ResNet")
    parser.add_argument("--res56", action = 'store_true', default = False, help = "Run 56-layer ResNet")

    args = parser.parse_args(argv)

    datasets = []
    architectures = []

    if args.mnist or args.all_data:
        datasets.append(MNIST())

    if args.cifar10 or args.all_data:
        datasets.append(CIFAR10())

    if len(datasets) == 0:
        print("No datasets specified, running with MNIST only")
        datasets.append(MNIST())

    # n.b. when we add an architecture we're actually adding callables that
    #      accept two arguments: dataset, scaling
    #      a.k.a. factory functions

    if args.conv or args.all_arch:
        architectures.append(CONV)

    if args.dropout or args.all_arch:
        architectures.append(lambda d, s: CONV(d, s, dropout = True))

    if args.nin or args.all_arch:
        architectures.append(NIN)

    if args.vgg or args.all_arch:
        architectures.append(VGG)

    if args.res20 or args.all_arch:
        architectures.append(lambda d, s: RES(d, s, depth = 20))

    if args.res32 or args.all_arch:
        architectures.append(lambda d, s: RES(d, s, depth = 32))

    if args.res56 or args.all_arch:
        architectures.append(lambda d, s: RES(d, s, depth = 56))

    if len(architectures) == 0:
        print("No architectures specified, running with simple convnet only")
        architectures.append(CONV)

    g = tf.config.list_physical_devices("GPU")

    for data, arch_factory in itertools.product(datasets, architectures):

        # instantiate the architecture with the dataset and scaling
        # why do we put the scaling in the arch rather than the dataset?
        # partly as then the arch is responsible for just adding Upsampling2D layers
        # without us having to manipulate the source data
        arch = arch_factory(data, args.scaling)

        root_filename = arch.name
        print(root_filename)

        model_filename = root_filename + ".model.h5"
        summary_filename = root_filename + ".summary"

        train = True
        if os.path.exists(model_filename) and os.path.exists(summary_filename) and not args.restart:
            print("Loading saved model from {}".format(model_filename))

            model = tf.keras.models.load_model(model_filename)

            with open(summary_filename) as f:
                summary = json.load(f)

            # if we're re-loading the model then we don't want to train unless
            # the --resume switch has been passed
            if not args.resume:
                train = False
        else:
            model = arch.build_model()

            if args.sgd:
                # in the past the "adam" optimiser has been quicker but for some models
                # we see no movement at all with that so the more consistently reliable
                # SGD is used
                opt = keras.optimizers.SGD(momentum = 0.1)
            else:
                opt = "adam"

            model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = [ "accuracy" ])

            summary = {
                "data": data.name,
                "arch": arch.name,
                "n_parameters": model.count_params()
            }

        summary["class_labels"] = data.class_labels
        summary["gpus"] = g

        if train:
            # callbacks looking for an improvement of at least min_delta in training loss
            min_delta = 0.0001
            reduce = keras.callbacks.ReduceLROnPlateau(monitor = "loss", patience = 3, verbose = 1, factor = 0.5, min_delta = min_delta)
            stopping = keras.callbacks.EarlyStopping(monitor = "loss", patience = 7, min_delta = min_delta)

            h = model.fit(data.x_train, data.y_train, epochs = args.epochs, validation_split = 0.15, callbacks = [ reduce, stopping ])
            print(h.history)  # loss, accuracy, val_loss, val_accuracy

            for key, values in h.history.items():
                # Keras likes to give us back a standard python list containing np.float32/64 which aren't
                # very JSON friendly, so convert to plain Python float here
                x = [ float(v) for v in values ]

                if key in summary:
                    summary[key].extend(x)
                else:
                    summary[key] = x

            model.save(model_filename)

        score = model.evaluate(data.x_test, data.y_test)
        print("Testing:", score)

        j_test = np.argmax(data.y_test, axis = 1)

        #TODO summarise prediction accuracy per-class
        y_predicted = model.predict(data.x_test)
        j_predicted = np.argmax(y_predicted, axis = 1)

        count_correct = 0
        confusion = np.zeros(shape = (data.n_classes, data.n_classes))

        for a, b in zip(j_test, j_predicted):
            confusion[a, b] += 1

            if a == b:
                count_correct += 1

        for i in range(data.n_classes):
            confusion[i, :] /= np.sum(confusion[i, :])

        confusion = np.round(confusion, 3)

        print(count_correct / data.n_test)

        print(confusion)  # .tolist() for JSON
        # plt.matshow(confusion, cmap = "binary")
        # plt.show()

        summary["test_accuracy"] = count_correct / data.n_test
        summary["confusion"] = confusion.tolist()

        with open(summary_filename, "w") as f:
            json.dump(summary, f, sort_keys = True, indent = 4)

        plot_model(model, to_file = root_filename + '.png', show_layer_names = False, show_shapes = True)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    print("done.")