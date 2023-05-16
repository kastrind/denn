import * as math from 'mathjs';
import { DataSet } from './DataSet';
import { Denn } from './Denn';
import { Utils } from './Utils';

const projectName = 'Test3';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const label2Sentences = require(`./assets/${projectName}/label2Sentences.json`);

// Load model from file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

//let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
//let queries = ["mother", "chimney", "boil"];
let queries = ["green bikes", "green bicycles", "blue pigs", "green cows"];
//let queries = ["owl", "dog", "cow"];
//let queries = ["open account", "setup account", "cancel account"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(' ');
  let queryTermEmbedding = [];
  let answer;
  let answers = [];
  let prevQueryTerms = [];
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn.predict([queryTermEmbedding]);
      answers.push(answer);
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(`Answer: ${label2Sentences[DataSet.mode(answers)]}`);
  }
});
