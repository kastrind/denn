import * as math from 'mathjs';
import { DataSet } from './DataSet';
import { Denn } from './Denn';
import { Utils } from './Utils';

const projectName = 'Test2';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const sentences = require(`./assets/${projectName}/sentences.json`);
//const onehot_to_labels = require(`./assets/${projectName}/onehot2labels.json`);
const binary2labels = require(`./assets/${projectName}/binary2labels.json`);

let label2Sentences = {};

// Form the label 2 sentences mapping
sentences.sentences.forEach((sentence, sIdx) => {
	sentence = sentence.toLowerCase();
	let sentenceLabel = "s"+sIdx;
	label2Sentences[sentenceLabel] = sentence;
});

// Load model from a file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

//let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
let queries = ["mother", "chimney", "boil"];
//let queries = ["owl", "dog", "cow"];
//let queries = ["open account", "setup account", "cancel account"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(' ');
  let queryTermEmbedding = [];
  let answerBinary;
  let answerLabel;
  let answer;
  let answers = [];
  let prevQueryTerms = [];
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      //answer = nn.predict([queryTermEmbedding], onehot_to_labels);
      answer = nn.predict([queryTermEmbedding]);
      // console.log(answer);
      //console.log(term);
      answerBinary = nn.output[0].toBinary(0.1).join('');
      answerLabel = binary2labels[answerBinary];
      answers.push(answerLabel);
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(`Answer: ${label2Sentences[DataSet.mode(answers)]}`);
  }
});
