import * as math from 'mathjs';
import { Activation } from './Activation';
import { Utils } from './Utils';
import fs from 'fs';

/**
 * Denn: Deep Neural Network
 */
export class Denn {

    constructor(X, Y, formation, learning_rate, activation_function=Activation.sigmoid, outputEncoding='NONE', encoding_to_label_map={}, initialized=false) {
        if (!Denn.checkXY(X, Y)) { return null; }
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
        this.outputEncoding = outputEncoding === 'ONEHOT' || outputEncoding === 'BINARY' ? outputEncoding : 'NONE';
        this.encoding_to_label_map = encoding_to_label_map;
        this.initLayers();
    }

    /**
     * Deserializes a model from a file in the given path.
     * @param {String} path  The path.
     */
    static deserialize(path) {
        console.log("Deserializing model from "+path+"...");
        let nn_raw = fs.readFileSync(path);
        let nn_deserialized = JSON.parse(nn_raw);
        let activations = {"sigmoid": Activation.sigmoid, "relu": Activation.relu};
        var nn = new Denn(nn_deserialized.X,
                          nn_deserialized.Y,
                          nn_deserialized.formation,
                          nn_deserialized.learning_rate,
                          activations[nn_deserialized.activationName],
                          nn_deserialized.outputEncoding,
                          nn_deserialized.encoding_to_label_map,
                          true);
        nn.input = nn_deserialized.input;
        nn.output = nn_deserialized.output;
        nn.layers = nn_deserialized.layers;
        console.log("Deserialized model successfully.");
        return nn;
    }

    // initializes the layers
    initLayers() {
        if (this.initialized === true) return;
        for (var i=0; i<this.formation.length; i++) {
            if (i==0) {
                this.layers.push({"weights": math.random([this.input[0].length, this.formation[i].neurons], 0, 0.001), "layer": [], "weights_copy": [], "layer_copy": [], "dropped_out_i": []});
            }else if (i<=this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i-1].neurons, this.formation[i].neurons], 0, 0.001), "layer": [], "weights_copy": [], "layer_copy": [], "dropped_out_i": []});
            } 
            if (i==this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i].neurons, this.Y[0].length], 0, 0.001), "weights_copy": []});
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

    backprop(epoch_idx, epochs) {
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
            this.layers[i].weights = math.add(this.layers[i].weights, math.multiply(math.transpose(this.layers[i-1].layer), math.multiply(this.layers[i].delta, math.max(0.1, this.learning_rate*(1-epoch_idx/epochs)))));
        }
        this.layers[0].weights = math.add(this.layers[0].weights, math.multiply(math.transpose(this.input), math.multiply(this.layers[0].delta, math.max(0.1, this.learning_rate*(1-epoch_idx/epochs)))));
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

    /**
     * Trains model.
     * @param {Integer} epochs Epochs
     * @param {Integer} batch_size The batch size 
     * @param {Float} error_threshold Error threshold below which early stopping triggers.
     * @param {Boolean} Y Verbosity.
     */
    train(epochs, batch_size, error_threshold, verbose) {
        console.log("\nTRAINING - start");
        let X = this.input;
        let y = this.Y;
        let epoch_mean_error = 0;
        let early_stopping_tolerance = Math.min(3, Math.floor(epochs/10));
        let early_stopping_cnt = 0;
        let early_stopping_tolerance_low_error_drop = Math.min(10, Math.floor(epochs/10));
        let early_stopping_cnt_low_error_drop = 0;
        let epoch_mean_error_prev = 1;

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
                    this.backprop(i, epochs);
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
                console.log("epoch: "+(i+1)+" / "+epochs+", mean error: "+epoch_mean_error);
            }
            // stop early on consecutive errors below error_threshold
            if (epoch_mean_error <= error_threshold) {
                early_stopping_cnt++;
            }else {
                early_stopping_cnt = 0;
            }
            if (early_stopping_cnt == early_stopping_tolerance) {
                if (verbose) console.log("Early stopping to avoid over-fitting.");
                break;
            }
            if (epoch_mean_error_prev - epoch_mean_error < 0.01*this.learning_rate/epochs ||
                epoch_mean_error_prev - epoch_mean_error < 0) {
                //early_stopping_cnt_low_error_drop++;
            }else {
                early_stopping_cnt_low_error_drop = 0;
            }
            if (early_stopping_cnt_low_error_drop == early_stopping_tolerance_low_error_drop) {
                if (verbose) console.log("Early stopping to avoid over-fitting.");
                break;
            }
            batch_cnt = 0;
            epoch_mean_error_prev = epoch_mean_error;
            epoch_mean_error = 0;
        }
        this.X = this.input = X;
        this.Y = this.output = y;
        console.log("TRAINING - end\n");
    }

    /**
     * Predicts output for a given tuple of 
     * @param {Array} input The input features array.
     */
    predict(input) {
        if (!input || !input.length) {
            console.error("Error: no input for prediction.");
            return null;
        }
        this.input = input;
        this.feedforward();
        //console.log(this.input);
        //console.log(this.output);
        //return this.output;
        let that = this;
        let output = [];

        if (this.outputEncoding === 'ONEHOT') {
            let onehot_array = [];
            let max_i = -1;
            this.output.forEach(function (row, i, arr) {
                onehot_array = math.zeros(row.length)._data;
                max_i = that.maxIndex(row);
                onehot_array[max_i] = 1;
                output.push(that.encoding_to_label_map[onehot_array.join('')]);
            });

        }else if (this.outputEncoding === 'BINARY') {
            this.output.forEach(function (row, i, arr) {
                output.push(that.encoding_to_label_map[row.toBinary(0.1).join('')]);
            });

        }else { // this.outputEncoding === 'NONE'
            return this.output;
        }
        return output;
    }

    /**
     * Tests model with a test set.
     * @param {Array} X The input features array.
     * @param {Array} Y The output variables array.
     * @param {Boolean} Y Verbosity.
     */
    test(X, Y, verbose) {
        if (!Denn.checkXY(X, Y)) { return null; }
        this.input = X;
        this.feedforward();
        //let output = this.predict(X);
        let hits = 0;
        let success=false;
        let report = "\nTESTING - start\n";
        let that = this;
        this.output.forEach(function (row, i, arr) {
            if (that.outputEncoding === 'ONEHOT') {
                if (that.maxIndex(row) === that.maxIndex(Y[i])) {
                    success=true;
                    hits++;
                }
            }else if (that.outputEncoding === 'BINARY') {
                if (row.toBinary(0.1).join('') === Y[i].join('')) {
                    success=true;
                    hits++;
                }
            }else { // that.outputEncoding === 'NONE'
                if (row.join('') === Y[i].join('')) {
                    success=true;
                    hits++;
                }
            }
            if (verbose) { report += "expected: "+Y[i]+", actual: "+row+", successful: "+success+", current hits: "+hits+"\n"; }
            success=false;
        });
        let accuracy = math.divide(hits, this.output.length);
        if (verbose) { report += "accuracy: "+accuracy+"\nTESTING - end\n"; console.log(report); }
        return accuracy;
    }

    // returns the index where the maximum element lies in the given array
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

    /**
     * Serializes the model to a file in the given path.
     * @param {String} path  The path.
     * @param {Boolean} finalize  If true, no further training can be resumed because input, output, X and Y are emptied
     */
    serialize(path, finalize=false) {
        console.log("Serializing model to "+path+"...");
        if (finalize) {
            this.input = [[]];
            this.output = 0;
            this.X = [[]];
            this.Y = [[]];
        }
        let nn_serialized = JSON.stringify(this);
        fs.writeFileSync(path, nn_serialized);
        console.log("Serialized model successfully.");
    }

    // error handling for input features and output arrays
    static checkXY(X, Y) {
        if (!X.length) {
            console.error("Error: empty training set.");
            return false;
        }
        else if (!Y ||  (X.length != Y.length) ) {
            console.error("Error: input and output variables have different cardinality.");
            return false;
        }
        return true;
    }


}