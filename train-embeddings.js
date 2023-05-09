import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Misc';
const dimensions = 20;
let embeddings = new Embeddings(`assets/${projectName}/corpus.txt`, `assets/${projectName}/sentences.json`, dimensions);

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 1, epochs = 1000, batch_size = 5, error_threshold = 0.015, verbose = true;
embeddings.train(formation, learning_rate, Activation.sigmoid, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
