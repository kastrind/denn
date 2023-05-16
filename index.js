import {Denn} from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';

// Import dataset
let dataset =  DataSet.import("assets/iris.txt", ',');

// Shuffle dataset
dataset = DataSet.shuffle(dataset);

// Separate input features from output variables and get a mapping of one-hot representation to the categorical values of the output
let onehot_to_labels = {};
let datasetXY = DataSet.separateXY(dataset, 4, 'ONEHOT', onehot_to_labels);

// Normalize dataset
DataSet.normalize(datasetXY.X);

// Keep 20% of the dataset for testing and the rest 80% for training
let train_test = DataSet.separateTrainTest(datasetXY, 0.2);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;

// This DNN will comprise one hidden layer of 5 neurons without drop-out probability
let formation = [{"neurons": 45, "dropout": 0.0}];
let learning_rate = 1.5;

// Instantiate DNN with a training set, architecture, learning rate, activation function of its hidden layer(s), one-hot encoding for the output and its mapping to the labels
var nn = new Denn(X, Y, formation, learning_rate, Activation.sigmoid, 'ONEHOT', onehot_to_labels);

// Train DNN
let epochs = 1000, batch_size = 3, error_threshold = 0.005, verbose = true;
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
let serialization_path = './nn-model.json';
nn.serialize(serialization_path);

// Load model from a file
var nn2 = Denn.deserialize(serialization_path);

// Resume training
nn2.train(epochs, batch_size, error_threshold, verbose);

// Test model
nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);
