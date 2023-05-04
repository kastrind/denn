import {Denn} from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';

// Import dataset
let dataset =  DataSet.import("assets/custom2.txt", ',');


let datasetXY = { X:[], Y:[] };

dataset.forEach(row => {
  datasetXY.X.push(row.slice(0, 5));
  datasetXY.Y.push(row.slice(5, 10));
});

let X = datasetXY.X;
let Y = datasetXY.Y;

// This DNN will comprise one hidden layer of 5 neurons without drop-out probability
let formation = [{"neurons": 92, "dropout": 0.0}];
let learning_rate = 0.8;

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
var nn = new Denn(X, Y, formation, learning_rate, Activation.sigmoid);

// Train DNN
let epochs = 50, batch_size = 3, error_threshold = 0.015, verbose = true;
nn.train(epochs, batch_size, error_threshold, verbose);

// Test model
//nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);
nn.test(X, Y, true);

