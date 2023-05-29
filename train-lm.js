import fs from 'fs';
import { DataSet } from './DataSet';
import { Activation } from './Activation';
import { Denn } from './Denn';

const projectName = 'Test';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const dimensions = embeddings.dimensions;
const labelSeparator = ';';

// Load and cleanse corpus
let corpusInitial = DataSet.loadCorpus(`./assets/${projectName}/corpus.txt`);
let corpus;
if (labelSeparator) {
  corpus = corpusInitial.replace(new RegExp(labelSeparator+"[^"+labelSeparator+"]+\r?\n", 'g'), '.\n');
  corpus = DataSet.cleanCorpus(corpus);
}else {
  corpus = DataSet.cleanCorpus(corpusInitial);
}

// Find sentences
let sentences = [];
sentences = corpus.split(/[.]/); // . marks sentences
sentences = sentences.filter(sentence => sentence.length);

let labels;
if (labelSeparator) {
  labels = corpusInitial.match(new RegExp(labelSeparator+"[^"+labelSeparator+"]+\r?\n", 'g'));
  labels.forEach((label, idx, arr) => { arr[idx] = label.replace(/[;\s.]/g, ''); });
  //console.log(labels);
}

// Save sentences
let corpusSentences = JSON.stringify({ sentences: sentences }, null, 2);
fs.writeFileSync(`./assets/${projectName}/sentences.json`, corpusSentences);

let trainSet = [];
let label2Sentences = {};

// Form the training set from each embedding for each term for each sentence in the corpus
sentences.forEach((sentence, sIdx) => {
  sentence = sentence.toLowerCase();
  let terms = sentence.split(' ');
  let sentenceLabel = labels ? labels[sIdx] : 's'+sIdx;
  label2Sentences[sentenceLabel] = sentence;
  for (let i=0; i<terms.length; i++) {
    if (Math.round(embeddings.maxFrequency / embeddings.dictionary[terms[i]]) > 1) {
      trainSet.push(embeddings.dictionaryEmbeddings[terms[i]].concat(sentenceLabel)); 
    }
  }
});

// Save label to sentence map
fs.writeFileSync(`./assets/${projectName}/label2Sentences.json`, JSON.stringify(label2Sentences, null, 2));

// Shuffle the dataset
trainSet = DataSet.shuffle(trainSet);

// Populate encoding to label map
let encoding_to_label_map = {};
let datasetXY = DataSet.separateXY(trainSet, dimensions, 'ONEHOT', encoding_to_label_map);

let X = datasetXY.X;
let Y = datasetXY.Y_encoded;

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 0.1, epochs = 300, batch_size = 50, error_threshold = 0.02, verbose = true;

// Instantiate DNN with a training set, architecture, learning rate, activation function of its hidden layer(s), one-hot/binary encoding for the output and its mapping to the labels
let nn = new Denn(X, Y, formation, learning_rate, Activation.relu, Activation.softPlus, 'ONEHOT', encoding_to_label_map);
// Train DNN
nn.train(epochs, batch_size, error_threshold, true, verbose);

// Save model to file, finalize it
nn.serialize(`./assets/${projectName}/lm.json`, true);

// Test model
//nn.test(X, Y, true);
