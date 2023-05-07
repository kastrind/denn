import * as math from 'mathjs';
import { Denn } from './Denn';

let embeddings = require('./embeddings.json');

// Load model from a file
var nn = Denn.deserialize('./lm.json');

let queries = ["alpha gamma", "one two four", "black green white crimson"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(" ");
  let queryTermEmbedding = [];
  let answer;
  let answers = [];
  let prevQueryTerms = [];
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn.predict([queryTermEmbedding]);
      console.log(Math.ceil(math.max(nn.output[0])/0.2));
      for (let c=0; c<Math.ceil(math.max(nn.output[0])/0.2); c++) { answers.push(answer[0]); }
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(answers);
    console.log(`Answer: ${label2Sentences[mode(answers)]}`);
  }

});

function mode(arr){
  return arr.sort((a,b) =>
        arr.filter(v => v===a).length
      - arr.filter(v => v===b).length
  ).pop();
}
