import {Denn} from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';

// Import dataset
let dataset =  DataSet.import("assets/iris.txt", ',');

// Shuffle dataset
dataset = DataSet.shuffle(dataset);

// Separate input features from output variables and get a mapping of one-hot representation to the categorical values of the output
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

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
var nn = new Denn(X, Y, formation, learning_rate, Activation.relu);

// Train DNN
let epochs = 100, batch_size = 1, error_threshold = 0.02, verbose = true;
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
let serialization_path = './nn-model.json';
nn.serialize(serialization_path);

// Load model from a file
var nn2 = Denn.deserialize(serialization_path);

// Resume training
nn2.train(epochs*4, batch_size, error_threshold, verbose);

// Test model
nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);

/*
Predictions below should be
  'Iris-versicolor',
  'Iris-setosa',
  'Iris-virginica',
  'Iris-virginica',
  'Iris-setosa',
console.log(nn2.predict([
    [ 0.7468354430379747, 0.6818181818181818, 0.6086956521739131, 0.6 ],
    [ 0.6075949367088607, 0.7727272727272726, 0.27536231884057966, 0.08 ],
    [ 0.810126582278481, 0.6136363636363636, 0.7681159420289855, 0.76 ],
    [ 0.7974683544303797, 0.7727272727272726, 0.8115942028985507, 0.96 ],
    [ 0.6962025316455696, 0.9545454545454545, 0.20289855072463767, 0.08 ]
], onehot_to_labels));
*/
