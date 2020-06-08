import * as math from 'mathjs';
import { Activation } from './Activation';
import fs from 'fs';

/*
TODO:
    -fix formation arg, OK
    -add learning rate, OK
    -rename structure, X
    -add epochs, OK
    -add predict func, OK
    -move activation function to its own class, OK
    -convert to ES2015, OK
    -dropout, OK
    -early stopping,
    -serialization, OK
    -accuracy measure, OK
    -batch size OK
    -map output labels
    -error handling
    -rename
    -import OK
    -train test OK
*/
export class NeuralNetwork {

    constructor(X, Y, formation, learning_rate, activation_function, initialized) {
        this.input = X;
        this.X = X;
        this.Y = Y;
        this.output = 0;
        this.layers = [];
        this.formation = formation;
        this.formationReversed = formation.slice().reverse();
        this.learning_rate = learning_rate;
        this.activation = activation_function;
        this.activationName = activation_function.name;
        this.initLayers();
    }

    static deserialize(path) {
        let nn_raw = fs.readFileSync(path);
        let nn_deserialized = JSON.parse(nn_raw);
        let activations = {"sigmoid": Activation.sigmoid, "relu": Activation.relu};
        var nn = new NeuralNetwork(nn_deserialized.X, nn_deserialized.Y, nn_deserialized.formation, nn_deserialized.learning_rate, activations[nn_deserialized.activationName], true);
        nn.input = nn_deserialized.input;
        nn.output = nn_deserialized.output;
        nn.layers = nn_deserialized.layers;
        return nn;
    }


    initLayers() {
        if (this.initialized === true) return;
        for (var i=0; i<this.formation.length; i++) {
            if (i==0) {
                this.layers.push({"weights": math.random([this.input[0].length, this.formation[i].neurons]), "layer": [], "weights_copy": [], "layer_copy": [], "dropped_out_i": []});
            }else if (i<=this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i-1].neurons, this.formation[i].neurons]), "layer": [], "weights_copy": [], "layer_copy": [], "dropped_out_i": []});
            } 
            if (i==this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i].neurons, this.Y[0].length]), "weights_copy": []});
            }
        }
        this.initialized = true;
        //this.printLayers();
    }
    
    feedforward() {
        for (var i=0; i<this.layers.length; i++) {
            if(i==0) { this.layers[i].layer = math.multiply(this.input, this.layers[0].weights).map(v => this.activation(v, false)); }
            else if (i<this.layers.length-1) { this.layers[i].layer = math.multiply(this.layers[i-1].layer, this.layers[i].weights).map(v => this.activation(v, false)); }
            if(i==this.layers.length-1) { this.output = math.multiply(this.layers[i-1].layer, this.layers[i].weights).map(v => Activation.sigmoid(v, false)); }
        }
    }

    backprop() {
        for (var i=this.layers.length-1; i>=0; i--) {
            if (i==this.layers.length-1) {
                this.layers[i].error = math.subtract(this.Y, this.output);
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.output.map(v => Activation.sigmoid(v, true)));
            }
            else if (i<this.layers.length-1 || i==0) {
                this.layers[i].error = math.multiply(this.layers[i+1].delta, math.transpose(this.layers[i+1].weights));
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.layers[i].layer.map(v => this.activation(v, true))) ;
            }
        }
        //update the weights
        for (var i=this.layers.length-1; i>=1; i--) {
            this.layers[i].weights = math.add(this.layers[i].weights, math.multiply(math.transpose(this.layers[i-1].layer), math.multiply(this.layers[i].delta, this.learning_rate)));
        }
        this.layers[0].weights = math.add(this.layers[0].weights, math.multiply(math.transpose(this.input), math.multiply(this.layers[0].delta, this.learning_rate)));
    }

    dropout() {
        let that = this;
        this.formation.forEach(function(layer, i) {
            that.dropoutLayer(i, layer.dropout);
        });
    }

    dropoutLayer(layer_i, p) {
        if (p==0) return;

        let dropped_out_i = [];
        //determine the indices of the layer to ignore based on given probability
        for (var i=0; i<this.layers[layer_i].weights[0].length; i++) {
            if (Math.random() <= p) {
                dropped_out_i.push(i);
            }
            if (dropped_out_i.length == this.layers[layer_i].weights[0].length -1) break;
        }

        dropped_out_i.sort(function(a, b){return b-a});
        this.layers[layer_i].dropped_out_i = dropped_out_i;

        //back-up the original layer
        this.layers[layer_i].weights_copy.push(JSON.parse(JSON.stringify(this.layers[layer_i].weights)));
        this.layers[layer_i].layer_copy = (JSON.parse(JSON.stringify(this.layers[layer_i].layer)));
        this.layers[layer_i+1].weights_copy.push(JSON.parse(JSON.stringify(this.layers[layer_i+1].weights)));
        
        let that = this;
        //perform the drop-out on the layer
        dropped_out_i.forEach(function(i) {
            that.layers[layer_i].weights.forEach( function(w){w.splice(i, 1);});
            that.layers[layer_i].layer.forEach( function(w){w.splice(i, 1);});
            that.layers[layer_i+1].weights.splice(i, 1);
        });
    }

    dropoutRestore() {
        let that = this;
        let layer_i = this.formation.length-1;
        this.formationReversed.forEach(function(layer) {
            if (layer.dropout>0) that.dropoutRestoreLayer(layer_i);
            layer_i -= 1;
        });
    }

    dropoutRestoreLayer(layer_i) {
        if (!this.layers[layer_i].dropped_out_i) return;

        let dropped_out_i = this.layers[layer_i].dropped_out_i;

        //update the original incoming weights of the given layer with the weights that were NOT dropped-out
        let last_w_c = this.layers[layer_i].weights_copy.pop();
        for (var i=0; i<last_w_c.length; i++) {
            for (var j=0; j<last_w_c[i].length; j++) {
                if (dropped_out_i.includes(j)) continue;
                else last_w_c[i][j] = this.layers[layer_i].weights[i].shift();  
            }
        }
        //update the original activations of the given layer with the ones that were NOT dropped-out
        let l_c = this.layers[layer_i].layer_copy;
        for (var i=0; i<l_c.length; i++) {
            for (var j=0; j<l_c[i].length; j++) {
                if (dropped_out_i.includes(j)) continue;
                else l_c[i][j] = this.layers[layer_i].layer[i].shift();  
            }
        }
        //update the original outgoing weights of the given layer  with the weights that were NOT dropped-out
        let last_w_c_next = this.layers[layer_i+1].weights_copy.pop();
        for (var j=0; j<last_w_c_next.length; j++) {
            if (dropped_out_i.includes(j)) continue;
            else last_w_c_next[j] = this.layers[layer_i+1].weights.shift();  
        }
        //restore the original network formation before the drop-out
        this.layers[layer_i].weights = JSON.parse(JSON.stringify(last_w_c));
        this.layers[layer_i].layer = JSON.parse(JSON.stringify(l_c));
        this.layers[layer_i+1].weights = JSON.parse(JSON.stringify(last_w_c_next));
    }

    printLayers() {
        for(var i=0; i<this.layers.length; i++) {
            console.log("LAYER "+i+":");
            console.log("WEIGHTS "+math.size(this.layers[i].weights)+":");
            for(var j=0; j<this.layers[i].weights.length; j++) {
                console.log(this.layers[i].weights[j]);
            }
            if (this.layers[i].layer) {
                console.log("ACTIVATIONS "+math.size(this.layers[i].layer)+":");
                console.log(this.layers[i].layer);
            } else console.log("ACTIVATIONS: "+this.layers[i].layer);
            
            console.log("END OF LAYER "+i);
        }
    }

    train(epochs, batch_size, verbose) {
        let report = "";
        let X = this.input;
        let y = this.Y;
        let epoch_mean_error = 0;
        for (var i=0; i<epochs; i++) {
            let start_i = 0, stop_i = 0;
            let batch_squared_errors = [], bse_size = [], batch_mean_error = 0;
            let batch_cnt = 0;
            for (var x=0; x<X.length; x++) {
                if (stop_i - start_i < batch_size) {
                    stop_i++;
                }
                if (stop_i - start_i == batch_size || x == X.length-1) {
                    batch_cnt++;
                    this.input = X.slice(start_i, stop_i);
                    this.Y = y.slice(start_i, stop_i);
                    if (i>0) this.dropout();
                    this.feedforward();
                    this.backprop();
                    if (i>0) this.dropoutRestore();

                    batch_squared_errors = math.subtract(this.Y, this.output);
                    batch_squared_errors.forEach(function(row, i, arr) { arr[i] = row.map(v => v*v); });
                    bse_size = math.size(batch_squared_errors);
                    batch_mean_error = math.divide(math.sum(batch_squared_errors), bse_size[0]*bse_size[1]);
                    epoch_mean_error += batch_mean_error;

                    start_i = stop_i;
                }
            }
            epoch_mean_error = math.divide(epoch_mean_error, batch_cnt);
            if (verbose) {
                console.log("epoch: "+i+" / "+epochs+", mean error: "+epoch_mean_error+"\n");
            }
            batch_cnt = 0;
            epoch_mean_error = 0;
        }
    }

    predict(input) {
        this.input = input;
        this.feedforward();
        return this.output;
    }

    test(X, Y, verbose) {
        let output = this.predict(X);
        let hits = 0;
        let report = "";
        let that = this;
        output.forEach(function (row, i, arr) {
            if (that.maxIndex(row) === that.maxIndex(Y[i])) {
                hits++;
            }
            if (verbose) { report += "expected: "+Y[i]+", actual: "+row+", current hits: "+hits+"\n"; }
        });
        let accuracy = math.divide(hits, output.length);
        if (verbose) { report += "accuracy: "+accuracy+"\n"; console.log(report); }
        return accuracy;
    }

    maxIndex(arr) {
        if (!arr.length) {
            return -1;
        }
        let max = arr[0];
        let max_i = 0;
        for (let i=0; i<arr.length; i++) {
            if (arr[i]>max) {
                max_i = i;
                max = arr[i]
            }
        }
        return max_i;
    }

    serialize(path) {
        let nn_serialized = JSON.stringify(this);
        fs.writeFileSync(path, nn_serialized);
    }

}