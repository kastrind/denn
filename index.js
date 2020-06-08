//var neuralnet = require('./NeuralNetwork');
import {NeuralNetwork} from './NeuralNetwork';
import { Activation } from './Activation';
import { DataSet } from './DataSet';

let ds = new DataSet();
let dataset =  DataSet.import("iris.txt", ',');
dataset = DataSet.shuffle(dataset);
let datasetXY = DataSet.separateXY(dataset, 4, true);
let train_test = DataSet.separateTrainTest(datasetXY, 0.2);

console.log(train_test.train.X.length+ " "+train_test.train.Y.length +" "+train_test.train.Y_one_hot.length);
console.log(train_test.test.X.length+ " "+train_test.test.Y.length +" "+train_test.test.Y_one_hot.length);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;
let formation = [{"neurons": 6, "dropout": 0.0}];
let learning_rate = 0.05;

var nn = new NeuralNetwork(X, Y, formation, learning_rate, Activation.sigmoid);
nn.train(1000, 3, true);
//console.log(nn.predict([[6.2,3.4,5.4,2.3]]));
let serialization_path = './nn-model.json';
nn.serialize(serialization_path);

var nn2 = NeuralNetwork.deserialize(serialization_path);
//nn2.train(100, 20, true);

nn2.test(train_test.test.X, train_test.test.Y_one_hot, true);