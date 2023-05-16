import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

const projectName = 'Test3';
const dimensions = 10;
let embeddings = new Embeddings(`assets/${projectName}/corpusEmbeddings.txt`, dimensions);

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 1, epochs = 1000, batch_size = 10, error_threshold = 0.02, verbose = true;
embeddings.train(formation, learning_rate, Activation.relu, epochs, batch_size, error_threshold, verbose);

embeddings.serialize(`./assets/${projectName}/embeddings.json`);
