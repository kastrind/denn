import fs from 'fs';
import { DataSet } from './DataSet';
import { Activation } from './Activation';
import { Denn } from './Denn';

const projectName = 'Test5';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const dimensions = embeddings.dimensions;

// Load and cleanse corpus
let corpus = DataSet.loadCorpus(`./assets/${projectName}/corpus.txt`);

// Find sentences
let sentences = [];
sentences = corpus.split(/[.]/); // . marks sentences
sentences = sentences.filter(sentence => sentence.length);

// Save sentences
let corpusSentences = JSON.stringify({ sentences: sentences }, null, 2);
fs.writeFileSync(`./assets/${projectName}/sentences.json`, corpusSentences);

let trainSet = [];
let label2Sentences = {};

// Form the training set from each embedding for each term for each sentence in the corpus
sentences.forEach((sentence, sIdx) => {
  sentence = sentence.toLowerCase();
  let terms = sentence.split(' ');
  let sentenceLabel = "s"+sIdx;
  label2Sentences[sentenceLabel] = sentence;
  for (let i=0; i<terms.length; i++) {
    if (Math.round(embeddings.maxFrequency / embeddings.dictionary[terms[i]]) > 1) {
      trainSet.push(embeddings.dictionaryEmbeddings[terms[i]].concat(sentenceLabel)); 
    }
  }
});

// Save label to sentence map
fs.writeFileSync(`./assets/${projectName}/label2Sentences.json`, JSON.stringify(label2Sentences, null, 2));

// Populate binary to label map
let binary_to_label_map = {};
let datasetXY = DataSet.separateXY(trainSet, dimensions, 'BINARY', binary_to_label_map);

console.log(binary_to_label_map);

// Shuffle the dataset
trainSet = DataSet.shuffle(trainSet);

let X = datasetXY.X;
let Y = datasetXY.Y_bin;

let formation = [{"neurons": 8, "dropout": 0.0},{"neurons": 8, "dropout": 0.0},{"neurons": 8, "dropout": 0.0},{"neurons": 8, "dropout": 0.0}];
let learning_rate = 2, epochs = 3000, batch_size = 20, error_threshold = 0.02, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate, activation function of its hidden layer(s), binary encoding for the output and its mapping to the labels
let nn = new Denn(X, Y, formation, learning_rate, Activation.relu, 'BINARY', binary_to_label_map);
// Train DNN
nn.train(epochs, batch_size, error_threshold, verbose);

// Save model to file, finalize it
nn.serialize(`./assets/${projectName}/lm.json`, true);

// Test model
nn.test(X, Y, true);
