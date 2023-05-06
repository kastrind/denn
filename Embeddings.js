import { Denn } from './Denn';
import fs from 'fs';

export class Embeddings {

  constructor(corpus, dimensions) {
    this.corpus = corpus;
    this.dimensions = dimensions;
    this.sentences = [];
    this.dictionary = {};
    this.dictionaryVectors = {};
    this.adjacencyPairs = [];
    this.trainSet = [];
    this.trainSetCSV = "";

    // clean-up corpus
    this.corpus = this.corpus.replace(/\s+/g, " "); // clear excess white-space
    this.corpus = this.corpus.replace(/(\)|\()/g, ""); // ignore parentheses
    this.corpus = this.corpus.replace(/[,]\s?/g, " "); // ignore commas
    this.corpus = this.corpus.replace(/[:]\s?/g, " "); // part : part as one sentence
    this.corpus = this.corpus.replace(/[.;]\s?/g, "."); // . ; treated the same
    // find sentences
    this.sentences = this.corpus.split(/[.]/); // . marks sentences
    this.sentences = this.sentences.filter(sentence => sentence.length);

    let adjacencyPairsForward = [];

    // build dictionary and adjacency pairs
    this.sentences.forEach(sentence => 
    {
      sentence = sentence.toLowerCase();
      let terms = sentence.split(" ");
      for (let i=0; i<terms.length; i++) {
        let term = terms[i];
        this.dictionary[term] = this.dictionary[term] ? this.dictionary[term]+1 : 1;
        for (let j=i; j<terms.length; j++) {
          if (j+1 < terms.length) {
            adjacencyPairsForward.push({x: term, y: terms[j+1]});
          }
        }
      }
    });

    let dictionarySize = Object.keys(this.dictionary).length;
    let buckets = Math.ceil(dictionarySize/this.dimensions);

    // generate initial vectors for each dictionary term
    Object.keys(this.dictionary).forEach((term, idx) =>
    {
      let termVector = Array.from({length: this.dimensions}, (x, i) => 0);
      let bucket = idx%buckets + 1;
      termVector[idx%this.dimensions] = bucket/buckets;
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
      this.trainSetCSV += row.join(',') + '\n';
      this.trainSet.push(row);
    });

  }

  train(formation, learning_rate, activation_function, epochs, batch_size, error_threshold, verbose = false) {
    let datasetXY = { X:[], Y:[] };

    this.trainSet.forEach(row => {
      datasetXY.X.push(row.slice(0, this.dimensions));
      datasetXY.Y.push(row.slice(this.dimensions, this.dimensions*2));
    });

    let X = datasetXY.X;
    let Y = datasetXY.Y;

    // Instantiate DNN with a training set, architecture, learning rate and activation function of its hidden layer(s)
    var nn = new Denn(X, Y, formation, learning_rate, activation_function);

    // Train DNN
    nn.train(epochs, batch_size, error_threshold, verbose);

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
        let embeddings_serialized = JSON.stringify(this.dictionaryEmbeddings, null, 2);
        fs.writeFileSync(path, embeddings_serialized);
        console.log("Serialized embeddings successfully.");
    }

  serializeAll(path) {
      console.log("Serializing embeddings to "+path+"...");
      let embeddings_serialized = JSON.stringify(this, null, 2);
      fs.writeFileSync(path, embeddings_serialized);
      console.log("Serialized embeddings successfully.");
  }

}
