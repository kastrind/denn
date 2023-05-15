import { Denn } from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';
import fs from 'fs';

const projectName = 'Test2';
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
    if (Math.floor(embeddings.maxFrequency / embeddings.dictionary[terms[i]]) > 1) {
      trainSet.push(embeddings.dictionaryEmbeddings[terms[i]].concat(sentenceLabel)); 
    }
  }
});

let binary_to_label_map = {};
let datasetXY = DataSet.separateXY(trainSet, dimensions, 'BINARY', binary_to_label_map);

fs.writeFileSync(`./assets/${projectName}/binary2labels.json`, JSON.stringify(binary_to_label_map, null, 2));

// Shuffle the dataset
trainSet = DataSet.shuffle(trainSet);

let X = datasetXY.X;
let Y = datasetXY.Y_bin;

let formation = [{"neurons": 22, "dropout": 0.0}];
let learning_rate = 1, epochs = 2000, batch_size = 20, error_threshold = 0.03, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
let nn = new Denn(X, Y, formation, learning_rate, Activation.relu, 'BINARY', binary_to_label_map);
// Train DNN
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to a file
nn.serialize(`./assets/${projectName}/lm.json`, false);

// Test model
nn.test(X, Y, true);
