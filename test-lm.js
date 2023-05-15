import * as math from 'mathjs';
import { Denn } from './Denn';

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

Array.prototype.toBinary = function(threshold) {
  for (let i=0; i < this.length; i++) {
    this[i] = this[i] >= threshold ? 1 : 0;
  }
  return this;
};

// Load model from a file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

//let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
let queries = ["mother", "chimney", "boil"];
//let queries = ["owl", "dog", "cow"];
//let queries = ["open account", "setup account", "cancel account"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(" ");
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
      console.log(answer);
      //console.log(term);
      answerBinary = nn.output[0].toBinary(0.1).join('');
      answerLabel = binary2labels[answerBinary];
      answers.push(answerLabel);
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(`Answer: ${label2Sentences[mode(answers)]}`);
  }
});

function mode(arr){
  return arr.sort((a,b) =>
        arr.filter(v => v===a).length
      - arr.filter(v => v===b).length
  ).pop();
};
