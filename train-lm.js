import { Denn } from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';
import fs from 'fs';

const projectName = 'Misc';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const dimensions = embeddings.dimensions;
const sentences = require(`./assets/${projectName}/sentences.json`);

let trainSet = [];
let label2Sentences = {};

// Form the training set from each embedding of each term of each sentence of the corpus
sentences.sentences.forEach((sentence, sIdx) => {
  sentence = sentence.toLowerCase();
  let terms = sentence.split(' ');
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

fs.writeFileSync(`./assets/${projectName}/onehot2labels.json`, JSON.stringify(onehot_to_labels, null, 2));

let train_test = DataSet.separateTrainTest(datasetXY, 0);

// Normalize dataset
DataSet.normalize(datasetXY.X);

let X = train_test.train.X;
let Y = train_test.train.Y_one_hot;

let formation = [{"neurons": 64, "dropout": 0.0},{"neurons": 16, "dropout": 0.0}];
let learning_rate = 1, epochs = 10000, batch_size = 30, error_threshold = 0.02, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
let nn = new Denn(X, Y, formation, learning_rate, Activation.relu);
// Train DNN
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
nn.serialize(`./assets/${projectName}/lm.json`);

nn.test(X, Y, true);
