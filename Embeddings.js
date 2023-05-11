import { Denn } from './Denn';
import fs from 'fs';
import { DataSet } from './DataSet';

export class Embeddings {

  constructor(corpusPath, sentencesOutPath, dimensions) {
    this.dimensions = dimensions;
    this.dictionary = {};
    this.dictionaryVectors = {};
    this.adjacencyPairs = [];
    this.trainSet = [];

    // load and cleanse corpus
    let corpus = Embeddings.loadCorpus(corpusPath);

    // find sentences
    let sentences = [];
    sentences = corpus.split(/[.]/); // . marks sentences
    sentences = sentences.filter(sentence => sentence.length);

    let corpusSentences = JSON.stringify({ sentences: sentences }, null, 2);
    fs.writeFileSync(sentencesOutPath, corpusSentences);

    let adjacencyPairsForward = [];

    // build dictionary and adjacency pairs
    sentences.forEach(sentence => 
    {
      sentence = sentence.toLowerCase();
      let terms = sentence.split(" ");
      for (let i=0; i<terms.length; i++) {
        let term = terms[i];
        this.dictionary[term] = this.dictionary[term] ? this.dictionary[term]+1 : 1;
        for (let j=i; j<i+3; j++) {
          if (j+1 < terms.length) {
            adjacencyPairsForward.push({x: term, y: terms[j+1]});
          }
        }
      }
    });

    this.maxFrequency = Math.max(...Object.values(this.dictionary));

    let dictionarySize = Object.keys(this.dictionary).length;
    let buckets = Math.ceil(dictionarySize/this.dimensions);

    // generate initial vectors for each dictionary term
    Object.keys(this.dictionary).forEach((term, idx) =>
    {
      let bucket = idx%buckets + 1;
      let termVector = Array.from({length: this.dimensions}, (x, i) => Math.random()*(bucket/(buckets*5)));
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
      if (Math.floor(this.maxFrequency / this.dictionary[pair.x]) > 1 &&
          Math.floor(this.maxFrequency / this.dictionary[pair.y]) > 1
      ) {
        this.trainSet.push(row);
      }
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
      let embeddings_serialized = JSON.stringify(this, null, 2);
      fs.writeFileSync(path, embeddings_serialized);
      console.log("Serialized embeddings successfully.");
  }

  static loadCorpus(path) {
    console.log("Loading corpus from "+path+"...");
    let corpus = fs.readFileSync(path).toString();
    // clean-up corpus
    console.log("Cleansing corpus...");
    corpus = corpus.replace(/[^A-Za-z\s.;:!?]/g, ""); // sanitize
    corpus = corpus.replace(/[,]\s?/g, ' '); // ignore commas
    corpus = corpus.replace(/[:]\s?/g, ' '); // part : part as one sentence
    corpus = corpus.replace(/[.;!?]\s?/g, '.'); // . ; ! ? treated the same
    corpus = corpus.replace(/\s+/g, ' '); // clear excess white-space
    console.log("Loaded corpus from "+path+"...");
    return corpus;
  }

}
