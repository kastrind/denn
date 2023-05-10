import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Test2';
const dimensions = 7;
let embeddings = new Embeddings(`assets/${projectName}/corpus.txt`, `assets/${projectName}/sentences.json`, dimensions);

let formation = [{"neurons": 92, "dropout": 0.01},{"neurons": 16, "dropout": 0.01}];
let learning_rate = 1, epochs = 100, batch_size = 5, error_threshold = 0.015, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
