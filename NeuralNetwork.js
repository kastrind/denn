import * as math from 'mathjs';

/*
TODO:
    -fix formation arg, OK
    -add learning rate, OK
    -rename structure, X
    -add epochs, OK
    -add predict func,
    -move activation function to its own class,
    -convert to ES2015, OK
    -dropout, OK
    -early stopping,
    -serialization,
    -accuracy measure,
    -batch size OK
*/
export class NeuralNetwork {

    constructor(X, Y, formation, learning_rate) {
        this.input = X;
        this.Y = Y;
        this.output = 0;
        this.layers = [];
        this.formation = formation;
        this.formationReversed = formation.slice().reverse();
        this.learning_rate = learning_rate;
        this.initLayers();
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

    sigmoid_d(z) {
        return math.dotMultiply(z, math.subtract(1, z));
    }
    
    feedforward() {
        for (var i=0; i<this.layers.length; i++) {
            if(i==0) { this.layers[i].layer = math.multiply(this.input, this.layers[0].weights).map(v => this.sigmoid(v, false)); }
            else if (i<this.layers.length-1) { this.layers[i].layer = math.multiply(this.layers[i-1].layer, this.layers[i].weights).map(v => this.sigmoid(v, false)); }
            if(i==this.layers.length-1) { this.output = math.multiply(this.layers[i-1].layer, this.layers[i].weights).map(v => this.sigmoid(v, false)); }
        }
    }

    backprop() {
        for (var i=this.layers.length-1; i>=0; i--) {
            if (i==this.layers.length-1) {
                this.layers[i].error = math.subtract(this.Y, this.output);
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.output.map(v => this.sigmoid_d(v)));
            }
            else if (i<this.layers.length-1 || i==0) {
                this.layers[i].error = math.multiply(this.layers[i+1].delta, math.transpose(this.layers[i+1].weights));
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.layers[i].layer.map(v => this.sigmoid_d(v))) ;
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
    
    sigmoid(z) {
        var bottom = math.add(1, math.exp(math.multiply(-1, z)));
        return math.dotDivide(1, bottom);
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

    train(epochs, batch_size) {
        let X = this.input;
        let y = this.Y;
        for (var i=0; i<epochs; i++) {
            let start_i = 0;
            let stop_i = 0;
            for (var x=0; x<X.length; x++) {
                if (stop_i - start_i < batch_size) {
                    stop_i++;
                }
                if (stop_i - start_i == batch_size || x == X.length-1) {
                    this.input = X.slice(start_i, stop_i);
                    this.Y = y.slice(start_i, stop_i);
                    if (i>0) this.dropout();
                    this.feedforward();
                    this.backprop();
                    if (i>0) this.dropoutRestore();
                    start_i = stop_i;
                }
            }
        }
    }
}