import { Activation } from './Activation';
import { Embeddings } from './Embeddings';

let dimensions = 5;
let embeddings = new Embeddings('assets/corpus.txt', 'assets/sentences.json', dimensions);

let formation = [{"neurons": 64, "dropout": 0.0}];
let learning_rate = 1, epochs = 10, batch_size = 1, error_threshold = 0.015, verbose = true;
embeddings.train(formation, learning_rate, Activation.sigmoid, epochs, batch_size, error_threshold, verbose);

embeddings.serialize('embeddings1.json');
