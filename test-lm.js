import * as math from 'mathjs';
import { Utils } from './Utils';
import { DataSet } from './DataSet';
import { Denn } from './Denn';

const projectName = 'Test';
const embeddings = require(`./assets/${projectName}/embeddings.json`);

// Load model from file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

//let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
//let queries = ["mother", "chimney", "boil"];
let queries = ["cancel card", "open a new account", "close my account", "card delivery", "what is my credit score", "how is the delivery of my card goind", "why is my credit score so low", "when will my card arrive", "close account"];
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
  let outputSum = math.zeros(Object.keys(nn.encoding_to_label_map).length)._data;
  let onehot_array = math.zeros(outputSum.length)._data;
  let max_i;
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn.predict([queryTermEmbedding]);
      //console.log(term);
      //console.log(nn.output[0]);
      outputSum = math.add(outputSum, nn.output[0]);
      //answers.push(answer);
      prevQueryTerms.push(term);
    }
  });
  //console.log(nn.encoding_to_label_map);
  //console.log(outputSum);
  max_i = nn.maxIndex(outputSum);
  onehot_array[max_i] = 1;
  console.log(nn.encoding_to_label_map[onehot_array.join('')]);
  if (answers.length) {
    console.log(`Answer: ${answers}`);
  }
});
