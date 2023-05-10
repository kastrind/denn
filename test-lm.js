import * as math from 'mathjs';
import { Denn } from './Denn';

const projectName = 'Test2';
const embeddings = require(`./assets/${projectName}/embeddings.json`);
const sentences = require(`./assets/${projectName}/sentences.json`);
const onehot_to_labels = require(`./assets/${projectName}/onehot2labels.json`);

let label2Sentences = {};

// Form the label 2 sentences mapping
sentences.sentences.forEach((sentence, sIdx) => {
	sentence = sentence.toLowerCase();
	let sentenceLabel = "s"+sIdx;
	label2Sentences[sentenceLabel] = sentence;
});

// Load model from a file
let nn = Denn.deserialize(`./assets/${projectName}/lm.json`);

let queries = ["wolf huffed", "house of straw", "house of bricks and cement"];
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
      answer = nn.predict([queryTermEmbedding], onehot_to_labels);
      for (let c=0; c<Math.ceil(math.max(nn.output[0])/0.2); c++) { answers.push(answer[0]); }
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
}
