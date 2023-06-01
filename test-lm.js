import * as math from 'mathjs';
import { Utils } from './Utils';
import { Denn } from './Denn';

const projectName = 'Test';
const embeddings = require(`./assets/${projectName}/embeddings.json`);

// Load model from file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

let queries = ["cancel card", "open a new account", "close my account", "card delivery", "what is my credit score", "how is the delivery of my card going", "why is my credit score so low", "when will my card arrive", "close account"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(' ');
  let queryTermEmbedding = [];
  let answer;
  let prevQueryTerms = [];
  let outputSum = math.zeros(Object.keys(nn.encoding_to_label_map)[0].length)._data;
  let outputAvg = math.zeros(Object.keys(nn.encoding_to_label_map)[0].length)._data;
  let onehot_array = math.zeros(outputSum.length)._data;
  let max_i;
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn.predict([queryTermEmbedding]);
      outputSum = math.add(outputSum, nn.output[0]);
      prevQueryTerms.push(term);
    }
  });
  if (nn.outputEncoding === 'ONEHOT') {
    max_i = nn.maxIndex(outputSum);
    onehot_array[max_i] = 1;
    console.log(nn.encoding_to_label_map[onehot_array.join('')]);
  }else if (nn.outputEncoding === 'BINARY') {
    outputAvg = math.dotDivide(outputSum, prevQueryTerms.length);
    console.log(nn.encoding_to_label_map[outputAvg.toBinary(nn.binaryOneConfidenceThreshold).join('')]);
  }
});
