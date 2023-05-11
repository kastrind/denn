import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'ThreeLittlePigs';
const dimensions = 30;
let embeddings = new Embeddings(`assets/${projectName}/corpus.txt`, `assets/${projectName}/sentences.json`, dimensions);

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 1, epochs = 10, batch_size = 20, error_threshold = 0.015, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
