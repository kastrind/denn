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

let queries = ["mother pig", "house of straw", "huffed and puffed", "boil in the fireplace", "enter through the chimney"];
//let queries = ["owl", "dog", "cow"];
//let queries = ["open account", "setup account", "cancel account"];
queries.forEach(query => {
  console.log(`Query: ${query}`);
  query = query.toLowerCase();
  let queryTerms = query.split(" ");
  let queryTermEmbedding = [];
  let answer;
  let answers = [];
  let runnerups = [];
  let answerConfs = {};
  let prevQueryTerms = [];
  let maxConf;
  queryTerms.forEach(term => {
    if (!prevQueryTerms.includes(term) && embeddings.dictionaryEmbeddings[term]) {
      queryTermEmbedding = embeddings.dictionaryEmbeddings[term];
      answer = nn.predict([queryTermEmbedding], onehot_to_labels);
      maxConf = math.max(nn.output[0]);
      if (maxConf >= 0.9) {
        answers.push(answer[0]);
      }
      answerConfs[answer] = answerConfs[answer] ? answerConfs[answer] + maxConf : maxConf;
      prevQueryTerms.push(term);
    }
  });
  if (answers.length) {
    console.log(`Answer (high conf.): ${label2Sentences[mode(answers)]}`);
  } else {
    answerConfs = Object.fromEntries(
      Object.entries(answerConfs).sort(([,a],[,b]) => b-a)
    );
    console.log(`Answer (max conf.): ${label2Sentences[Object.keys(answerConfs)[0]]}`);
  }

});

function mode(arr){
  return arr.sort((a,b) =>
        arr.filter(v => v===a).length
      - arr.filter(v => v===b).length
  ).pop();
}
