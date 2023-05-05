let corpus = "One two three. alpha beta gamma. The king was a wise man. The queen was a kind woman.";
let dimensions = 5;

corpus = corpus.replace(/[,]\s?/g, " "); // ignore commas
corpus = corpus.replace(/[:]\s?/g, " "); // part : part as one sentence
corpus = corpus.replace(/[.;]\s?/g, "."); // . ; treated the same
let sentences = corpus.split(/[.,]/); // . marks sentences
sentences = sentences.filter(sentence => sentence.length);

let adjacencyPairsForward = [];
let dictionary = {};

sentences.forEach(sentence => 
{
  sentence = sentence.toLowerCase();
	let terms = sentence.split(" ");
  for (let i=0; i<terms.length; i++) {
    let term = terms[i];
    dictionary[term] = dictionary[term] ? dictionary[term]+1 : 1;
    for (let j=i; j<terms.length; j++) {
      if (j+1 < terms.length) {
        adjacencyPairsForward.push({x: term, y: terms[j+1]});
      }
    }
  }
});

let dictionarySize = Object.keys(dictionary).length;
let dictionaryVectors = {};
let buckets = Math.ceil(dictionarySize/dimensions);
//console.log(buckets);

Object.keys(dictionary).forEach((term, idx) =>
{
  let termVector = Array.from({length: dimensions}, (x, i) => 0);
  let bucket = idx%buckets + 1;
  termVector[idx%dimensions] = bucket/buckets;
  dictionaryVectors[term] = termVector;
});
//console.log(Object.values(dictionaryVectors));
console.log(dictionaryVectors);

let adjacencyPairsBackward = [];
adjacencyPairsForward.forEach(pair => 
{
  adjacencyPairsBackward.push({x: pair.y, y: pair.x});
});

let adjacencyPairs = adjacencyPairsForward.concat(adjacencyPairsBackward);

let test = adjacencyPairs.filter(pair =>
{
  return pair.x == "man";
});

let trainSet = [];
let trainSetCSV = "";

adjacencyPairs.forEach(pair =>
{
	let row = dictionaryVectors[pair.x].concat(dictionaryVectors[pair.y]);
  trainSetCSV += row.join(',') + '\n'
  trainSet.push(row);
});

console.log(trainSetCSV);

//console.log(test);
//console.log(dictionary);

//console.log(adjacencyPairs);
//console.log(adjacencyPairs.length);