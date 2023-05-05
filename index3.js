import { Denn } from './Denn';
import { Activation } from './Activation';
import { DataSet } from './DataSet';
import { Embeddings } from './Embeddings';


let corpus = "One two three. alpha beta gamma. The king was a wise man. The queen was a kind woman.";
let dimensions = 5;
let formation = [{"neurons": 32, "dropout": 0.0}];
let learning_rate = 1.5, epochs = 400, batch_size = 3, error_threshold = 0.015, verbose = true;

let embeddings = new Embeddings(corpus, dimensions);
embeddings.train(formation, learning_rate, Activation.sigmoid, epochs, batch_size, error_threshold, verbose);
