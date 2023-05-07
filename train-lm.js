import { Denn } from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';

let dimensions = 5;
let embeddings = require('./embeddings.json');

let trainSet = [];
let label2Sentences = {};

// Form the training set from each embedding of each term of each sentence of the corpus
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

// Separate input features from output variables and get a mapping of one-hot representation to the categorical values of the output
let onehot_to_labels = {};
let datasetXY = DataSet.separateXY(trainSet, dimensions, true, onehot_to_labels);
console.log(onehot_to_labels);

let train_test = DataSet.separateTrainTest(datasetXY, 0);

// Normalize dataset
DataSet.normalize(datasetXY.X);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;

let formation = [{"neurons": 32, "dropout": 0.0}, {"neurons": 16, "dropout": 0.0}];
let learning_rate = 5, epochs = 10000, batch_size = 3, error_threshold = 0.02, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
var nn = new Denn(X, Y, formation, learning_rate, Activation.sigmoid);
// Train DNN
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
nn.serialize('./lm.json');

nn.test(X, Y, true);
