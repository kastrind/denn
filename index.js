//var neuralnet = require('./NeuralNetwork');
import {NeuralNetwork} from './NeuralNetwork';
import { Activation } from './Activation';
import { DataSet } from './DataSet';
import * as math from 'mathjs';

let ds = new DataSet();
let dataset =  DataSet.import("iris.txt", ',');
dataset = DataSet.shuffle(dataset);
let onehot_to_labels = {};
let datasetXY = DataSet.separateXY(dataset, 4, true, onehot_to_labels);
DataSet.normalize(datasetXY.X);
let train_test = DataSet.separateTrainTest(datasetXY, 0.2);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;
let formation = [{"neurons": 5, "dropout": 0.0}];
let learning_rate = 0.15;
var nn = new NeuralNetwork(X, Y, formation, learning_rate, Activation.relu);
nn.train(500, 1, 0.02, true);

let serialization_path = './nn-model.json';
nn.serialize(serialization_path);
var nn2 = NeuralNetwork.deserialize(serialization_path);
//nn2.train(100, 20, true);

nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);

console.log(nn2.predict([[0.9,0.5,0.3,0.2],[0.9,0.2,0.3,0.1],[0.5,0.5,0.5,0.4]], onehot_to_labels));
console.log(nn2.predict([[0.9,0.5,0.3,0.2],[0.9,0.2,0.3,0.1],[0.5,0.5,0.5,0.4]]));