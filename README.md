# :sparkler: DeNN: a Deep Neural Network in JavaScript
Neural Networks are a fascinating and even mysterious subject in Machine Learning.
This project implements a Deep Neural Network, a certain type of Neural Networks which employs at least one hidden layer of neurons.

# :wrench: Setup & Run
1. `npm install`
2. `babel-node index.js --presets=env`

:collision: If `babel-node` command is not recognized (may occur in Windows 10), please install globally: `npm install babel-cli -g`

# :airplane: Usage


    import {Denn} from './Denn';
    import { Activation } from './Activation';
    import { DataSet } from './DataSet';

    // Import dataset
    let dataset =  DataSet.import("iris.txt", ',');

    // Shuffle dataset
    dataset = DataSet.shuffle(dataset);

    // Separate input features from output variables and get a mapping of one-hot representation
    // to the categorical values of the output
    let onehot_to_labels = {};
    let datasetXY = DataSet.separateXY(dataset, 4, true, onehot_to_labels);

    // Normalize dataset
    DataSet.normalize(datasetXY.X);

    // Keep 20% of the dataset for testing and the rest 80% for training
    let train_test = DataSet.separateTrainTest(datasetXY, 0.2);

    let X = train_test.train.X;
    let Y = train_test.train.Y_one_hot;

    // This DNN will comprise one hidden layer of 5 neurons without drop-out probability
    let formation = [{"neurons": 5, "dropout": 0.0}];
    let learning_rate = 0.15;

    // Instantiate DNN with a training set, architecture, learning rate and activation function
    // of its hidden layer(s)
    var nn = new Denn(X, Y, formation, learning_rate, Activation.relu, 'ONEHOT', onehot_to_labels);

    // Train DNN
    let epochs = 100, batch_size = 1, error_threshold = 0.02, verbose = true;
    nn.train(epochs, batch_size, error_threshold, false, verbose);

    // Save model to a file
    let serialization_path = './nn-model.json';
    nn.serialize(serialization_path);

    // Load model from a file
    var nn2 = Denn.deserialize(serialization_path);

    // Resume training
    nn2.train(epochs*4, batch_size, error_threshold, false, verbose);

    // Test model
    nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);

# :crown: Acknowledgements
- [Machine Learning, Data Science and Deep Learning with Python - Frank Kane](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/)
- [Understanding the Mathematics behind Gradient Descent - Parul Pandey](https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e)
- [How to build your own Neural Network from scratch in Python - James Loy](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
- [Creating of neural network using JavaScript in 7 minutes! - Petro Liashchynskyi](https://dev.to/liashchynskyi/creating-of-neural-network-using-javascript-in-7minutes-o21)
- [Backpropagation calculus - Channel 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8&t=307s)
