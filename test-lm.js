import { Utils } from './Utils';
import { DataSet } from './DataSet';
import { Denn } from './Denn';

const projectName = 'Test5';
const embeddings = require(`./assets/${projectName}/embeddings.json`);

// Load model from file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

//let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
//let queries = ["mother", "chimney", "boil"];
let queries = ["i want to cancel lost card", "open a new account", "i want to close my account", "how is the delivery of my card", "cancel my deposit"];
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
    console.log(`Answer: ${answers}`);
  }
});
