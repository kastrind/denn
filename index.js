//var neuralnet = require('./NeuralNetwork');
import * as neuralnet from './NeuralNetwork';

// let X = [[0,0,1],
//      [0,1,1],
//      [1,0,1],
//      [1,1,1]];
let X = [[1,0,1],
        [0,1,1],
        [1,0,1],
        [1,0,0],
        [1,1,1],
        [1,1,0]];
let y = [[1],[0],[1],[0],[0],[0]];
let formation = [15];
var nn = new neuralnet.NeuralNetwork(X, y, formation);

nn.initLayers(formation);

//nn.train(500, X, y);
for (var i=0; i<5000; i++) {
  nn.dropout(0, 0.5);
  nn.feedforward();
  nn.backprop();
  nn.dropout_restore(0);
}
nn.printLayers();
console.log(nn.output);

// for (var i=0; i<500; i++) {
//   for (var x=0; x<X.length; x++) {
//   nn.input = [X[x]];
//   nn.Y = [y[x]];
//   nn.Y = [y[x]];
//   //nn.dropout(0, 0.05);
//   nn.feedforward();
//   nn.backprop();
//   console.log(nn.output);
//   //nn.dropout_restore(0);
//   }
// }
//nn.printLayers();


//predict:
// X = [[1,0,1]];
// nn.input = X;
// for (var i=0; i<1; i++) {
//   nn.feedforward();
// }
// console.log(nn.output)