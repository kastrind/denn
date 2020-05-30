//var neuralnet = require('./NeuralNetwork');
import {NeuralNetwork} from './NeuralNetwork';

let X = [[1,0,1],
        [0,1,1],
        [1,0,1],
        [1,0,0],
        [1,1,1],
        [1,1,0]];
let y = [[1],[0],[1],[0],[0],[0]];
let formation = [{"neurons": 5, "dropout":0.2},{"neurons": 3, "dropout":0.1}];
let learning_rate = 1;
var nn = new NeuralNetwork(X, y, formation, learning_rate);

nn.train(700, 6);

console.log(nn.predict([[0,0,0]]));


for (var i=0; i<0; i++) {
  //console.log("ITERATION i="+i+":");
  //if (i>0) nn.dropout(0, 0.2);
  //if (i>0) nn.dropout(1, 0.1);
  if (i>0) nn.dropout();
  nn.feedforward();
  //console.log("\nAFTER FF:\n");
  //nn.printLayers();
  nn.backprop();
  //console.log("\nRESTORING DROPOUTS:\n");
  //if (i>0) nn.dropout_restore(1);
  //if (i>0) nn.dropout_restore(0);
  if (i>0) nn.dropoutRestore();
  //console.log("\n\n\n==============\n\n\n");
  //nn.printLayers();
}
console.log(nn.output);

//predict:
// X = [[1,0,1]];
// nn.input = X;
// for (var i=0; i<1; i++) {
//   nn.feedforward();
// }
// console.log(nn.output)