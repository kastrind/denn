import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Test';
const dimensions = 40;
let embeddings = new Embeddings(`assets/${projectName}/embeddingsCorpus.txt`, dimensions);

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 0.05, epochs = 100, batch_size = 10, error_threshold = 0.001, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
