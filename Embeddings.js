import fs from 'fs';
import * as math from 'mathjs';
import { DataSet } from './DataSet';
import { Denn } from './Denn';

export class Embeddings {

  /**
   * @param {String} corpusPath path to corpus file
   * @param {Number} dimensions cardinality of dimensions
   * @param {Number} frequencyThreshold dictionary term frequency threshold 0-1 factor of max frequency above which term will be ignored (default: 1 - ignores none)
   * @param {Number} adjacencyRange how many terms are considered adjacent (default: 5)
   */
  constructor(corpusPath, dimensions, frequencyThreshold=1, adjacencyRange=5) {
    this.dimensions = dimensions;
    this.dictionary = {};
    this.dictionaryVectors = {};
    this.adjacencyPairs = [];
    this.trainSet = [];
    this.frequencyThreshold = frequencyThreshold;
    this.adjacencyRange = adjacencyRange;

    // load and cleanse corpus
    let corpus = DataSet.loadCorpus(corpusPath);
    corpus = DataSet.cleanCorpus(corpus);

    // find sentences
    let sentences = [];
    sentences = corpus.split(/[.]/); // . marks sentences
    sentences = sentences.filter(sentence => sentence.length);

    let adjacencyPairsForward = [];

    // build dictionary and adjacency pairs
    sentences.forEach(sentence => 
    {
      sentence = sentence.toLowerCase();
      let terms = sentence.split(" ");
      for (let i=0; i<terms.length; i++) {
        let term = terms[i];
        this.dictionary[term] = this.dictionary[term] ? this.dictionary[term]+1 : 1;
      }
    });

    this.maxFrequency = Math.max(...Object.values(this.dictionary));

    this.dictionary = Object.fromEntries(Object.entries(this.dictionary).filter(([term, freq]) => freq <= this.maxFrequency * this.frequencyThreshold));

    sentences.forEach(sentence => 
    {
      sentence = sentence.toLowerCase();
      let terms = sentence.split(" ");
      for (let i=0; i<terms.length; i++) {
        let term = terms[i];
        if (!this.dictionary[term]) continue;
        for (let j = i; j < i + this.adjacencyRange; j++) {
          if (j+1 < terms.length) {
            if (this.dictionary[terms[j+1]]) {
              let freqRatioX = this.dictionary[term] / this.maxFrequency;
              let freqRatioY = this.dictionary[terms[j+1]] / this.maxFrequency;
              if (freqRatioX <= this.frequencyThreshold && freqRatioY <= this.frequencyThreshold) {
                adjacencyPairsForward.push({x: term, y: terms[j+1]});
              }
            }
          }
        }
      }
    });

    let dictionarySize = Object.keys(this.dictionary).length;
    let buckets = Math.ceil(dictionarySize/this.dimensions);

    // generate initial vectors for each dictionary term
    Object.keys(this.dictionary).forEach((term, idx) =>
    {
      let freq = this.dictionary[term];
      let bucket = idx%buckets + 1;
      let termVector = Array.from({length: this.dimensions}, (x, i) =>(bucket/buckets)*math.random(0.01, 0.09));
      termVector[idx%this.dimensions] = (bucket/buckets)*math.random(0.9, 1.0);
      this.dictionaryVectors[term] = termVector;
    });

    let adjacencyPairsBackward = [];
    // build backward adjacency pairs
    adjacencyPairsForward.forEach(pair => 
    {
      adjacencyPairsBackward.push({x: pair.y, y: pair.x});
    });

    // set all adjacency pairs
    this.adjacencyPairs = adjacencyPairsForward.concat(adjacencyPairsBackward);

    // build the embeddings training set
    this.adjacencyPairs.forEach(pair =>
    {
      let row = this.dictionaryVectors[pair.x].concat(this.dictionaryVectors[pair.y]);
      this.trainSet.push(row);
    });

  }

  train(formation, learning_rate, activation_function, output_activation_function, epochs, batch_size, error_threshold, verbose = false) {
    let datasetXY = { X:[], Y:[] };

    this.trainSet.forEach(row => {
      datasetXY.X.push(row.slice(0, this.dimensions));
      datasetXY.Y.push(row.slice(this.dimensions, this.dimensions*2));
    });

    let X = datasetXY.X;
    let Y = datasetXY.Y;

    // Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
    var nn = new Denn(X, Y, formation, learning_rate, activation_function, output_activation_function);

    // Train DNN
    nn.train(epochs, batch_size, error_threshold, false, verbose);

    this.dictionaryEmbeddings = {};

    Object.keys(this.dictionaryVectors).forEach( term => {
      //console.log(term);
      //console.log(this.dictionaryVectors[term]);
      nn.predict([this.dictionaryVectors[term]]);
      this.dictionaryEmbeddings[term] = nn.output[0];
    });

    //console.log(this.dictionaryEmbeddings);
    return this.dictionaryEmbeddings;
  }

  /**
   * Serializes embeddings to a file in the given path.
   * @param {String} path  The path.
   */
  serialize(path) {
      console.log("Serializing embeddings to "+path+"...");
      let embeddings_serialized = JSON.stringify(this, null, 2);
      fs.writeFileSync(path, embeddings_serialized);
      console.log("Serialized embeddings successfully.");
  }

}
