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

/*
Shoud be
  'Iris-versicolor',
  'Iris-setosa',
  'Iris-virginica',
  'Iris-virginica',
  'Iris-setosa',
*/
console.log(nn2.predict([
    [ 0.7468354430379747, 0.6818181818181818, 0.6086956521739131, 0.6 ],
    [ 0.6075949367088607, 0.7727272727272726, 0.27536231884057966, 0.08 ],
    [ 0.810126582278481, 0.6136363636363636, 0.7681159420289855, 0.76 ],
    [ 0.7974683544303797, 0.7727272727272726, 0.8115942028985507, 0.96 ],
    [ 0.6962025316455696, 0.9545454545454545, 0.20289855072463767, 0.08 ]
], onehot_to_labels));
console.log(nn2.predict([
    [ 0.7468354430379747, 0.6818181818181818, 0.6086956521739131, 0.6 ],
    [ 0.6075949367088607, 0.7727272727272726, 0.27536231884057966, 0.08 ],
    [ 0.810126582278481, 0.6136363636363636, 0.7681159420289855, 0.76 ],
    [ 0.7974683544303797, 0.7727272727272726, 0.8115942028985507, 0.96 ],
    [ 0.6962025316455696, 0.9545454545454545, 0.20289855072463767, 0.08 ],
]));