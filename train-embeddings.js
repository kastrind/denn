import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Test2';
const dimensions = 20;
let embeddings = new Embeddings(`assets/${projectName}/corpus.txt`, `assets/${projectName}/sentences.json`, dimensions);

let formation = [{"neurons": 32, "dropout": 0.0}];
let learning_rate = 2, epochs = 200, batch_size = 10, error_threshold = 0.02, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
