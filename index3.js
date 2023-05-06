import * as math from 'mathjs';
import { Denn } from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';
import { Embeddings } from './Embeddings';


let corpus = "Alpha beta gamma delta. One two three four. Black brown green red white blue yellow.";
let dimensions = 5;
let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 1, epochs = 1000, batch_size = 1, error_threshold = 0.015, verbose = true;

/*
let embeddings = new Embeddings(corpus, dimensions);
embeddings.train(formation, learning_rate, Activation.sigmoid, epochs, batch_size, error_threshold, verbose);

embeddings.serializeAll('embeddings.json');
*/


let embeddings = require('./embeddings.json');


let trainSet = [];
let label2Sentences = {};

embeddings.sentences.forEach((sentence, sIdx) => {
  sentence = sentence.toLowerCase();
  let terms = sentence.split(" ");
  let sentenceLabel = "s"+sIdx;
  label2Sentences[sentenceLabel] = sentence;
  for (let i=0; i<terms.length; i++) {
    //console.log(terms[i]);
    trainSet.push(embeddings.dictionaryEmbeddings[terms[i]].concat([sentenceLabel])); 
  }
});

// Shuffle dataset
//trainSet = DataSet.shuffle(trainSet);

// Separate input features from output variables and get a mapping of one-hot representation to the categorical values of the output
let onehot_to_labels = {};
let datasetXY = DataSet.separateXY(trainSet, 5, true, onehot_to_labels);
console.log(onehot_to_labels);

let train_test = DataSet.separateTrainTest(datasetXY, 0);

// Normalize dataset
DataSet.normalize(datasetXY.X);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;

/*
// console.log(trainSet[0]);
// console.log(datasetXY.X[0]);
// console.log(datasetXY.Y[0]);
// console.log(train_test.train.X[0]);
// console.log(train_test.train.Y_one_hot);
// console.log(train_test.train.X.length);
// console.log(datasetXY.X.length);
// console.log(trainSet.length);

formation = [{"neurons": 32, "dropout": 0.0}, {"neurons": 16, "dropout": 0.0}];
learning_rate = 5, epochs = 10000, batch_size = 3, error_threshold = 0.02, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
var nn = new Denn(X, Y, formation, learning_rate, Activation.sigmoid);
// Train DNN
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
nn.serialize('./lm.json');

nn.test(X, Y, true);
*/

// Load model from a file
var nn2 = Denn.deserialize('./lm.json');
nn2.test(X, Y, true);

let queries = ["alpha gamma", "one two four", "black green white crimson"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(" ");
  let queryTermEmbedding = [];
  let answer;
  let answers = [];
  let prevQueryTerms = [];
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn2.predict([queryTermEmbedding], onehot_to_labels);
      console.log(Math.ceil(math.max(nn2.output[0])/0.2));
      for (let c=0; c<Math.ceil(math.max(nn2.output[0])/0.2); c++) { answers.push(answer[0]); }
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(answers);
    console.log(`Answer: ${label2Sentences[mode(answers)]}`);
  }

});

function mode(arr){
  return arr.sort((a,b) =>
        arr.filter(v => v===a).length
      - arr.filter(v => v===b).length
  ).pop();
}

