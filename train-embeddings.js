import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Test5';
const dimensions = 10;
let embeddings = new Embeddings(`assets/${projectName}/corpus.txt`, dimensions);

let formation = [{"neurons": 16, "dropout": 0.0}];
let learning_rate = 1, epochs = 300, batch_size = 20, error_threshold = 0.01, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
