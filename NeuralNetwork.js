import * as math from 'mathjs';

/*
TODO:
    -fix formation arg OK
    -add learning rate,
    -rename structure, X
    -add epochs
    -add predict func
    -move activation function to its own class
    -convert to ES2015 OK
    -dropout OK
    -early stopping
*/
export class NeuralNetwork {

    constructor(X, Y, formation) {
        this.input = X;
        this.Y = Y;
        this.output = 0;
        this.layers = [];
        this.formation = formation;
        this.initLayers();
    }


    initLayers() {
        if (this.initialized === true) return;
        for (var i=0; i<this.formation.length; i++) {
            if (i==0) {
                this.layers.push({"weights": math.random([this.input[0].length, this.formation[i]]), "layer": 0});
            }else if (i<this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i-1], this.formation[i]]), "layer": 0});
            } 
            if (i==this.formation.length-1) {
                this.layers.push({"weights": math.random([this.formation[i], this.Y[0].length])});
            }
        }
        this.initialized = true;
        //this.printLayers();
    }

    printLayers() {
        for(var i=0; i<this.layers.length; i++) {
            console.log("LAYER "+i+":");
            console.log("WEIGHTS:");
            for(var j=0; j<this.layers[i].weights.length; j++) {
                console.log(this.layers[i].weights[j]);
            }
            console.log("ACTIVATIONS:");
            console.log(this.layers[i].layer);
            console.log("END OF LAYER "+i);
        }
    }

    dropout(layer_i, p) {
        if (p==0) return;

        let dropped_out_i = [];
        for (var i=0; i<this.layers[layer_i].weights[0].length; i++) {
            if (Math.random() <= p) {
                dropped_out_i.push(i);
            }
            if (dropped_out_i.length == this.layers[layer_i].weights[0].length -1) break;
        }
        if (dropped_out_i.length==0) return;
        this.layers[layer_i].dropped_out_i = dropped_out_i;
        dropped_out_i.sort(function(a, b){return b-a});
        //console.log(dropped_out_i);

        this.layers[layer_i].weights_copy = JSON.parse(JSON.stringify(this.layers[layer_i].weights));
        this.layers[layer_i].layer_copy = this.layers[layer_i].layer ? JSON.parse(JSON.stringify(this.layers[layer_i].layer)) : 0;
        this.layers[layer_i+1].weights_copy = JSON.parse(JSON.stringify(this.layers[layer_i+1].weights));
        
        let that = this;
        dropped_out_i.forEach(function(i) {
            that.layers[layer_i].weights.forEach( function(w){w.splice(i, 1);});
            if (that.layers[layer_i].layer) {
                //that.layers[layer_i].layer.splice(i, 1);
                that.layers[layer_i].layer.forEach( function(w){w.splice(i, 1);});
            }
            that.layers[layer_i+1].weights.splice(i, 1);
        });

        //console.log(this.layers[layer_i])
        return dropped_out_i;
    }

    dropout_restore(layer_i) {
        if (!this.layers[layer_i].dropped_out_i) return;

        for (var i=0; i<this.layers[layer_i].weights_copy.length; i++) {
            for (var j=0; j<this.layers[layer_i].weights_copy[i].length; j++) {
                if (this.layers[layer_i].dropped_out_i.includes(j)) continue;
                else this.layers[layer_i].weights_copy[i][j] = this.layers[layer_i].weights[i].shift();  
            }
        }
        for (var i=0; i<this.layers[layer_i].layer_copy.length; i++) {
            for (var j=0; j<this.layers[layer_i].layer_copy[i].length; j++) {
                if (this.layers[layer_i].dropped_out_i.includes(j)) continue;
                else this.layers[layer_i].layer_copy[i][j] = this.layers[layer_i].layer[i].shift();  
            }
        }
        if (this.layers[layer_i].layer_copy) {
            for (var j=0; j<this.layers[layer_i].layer_copy.length; j++) {
                if (this.layers[layer_i].dropped_out_i.includes(j)) continue;
                else this.layers[layer_i].layer_copy[j] = this.layers[layer_i].layer.shift();  
            }
        }
        for (var j=0; j<this.layers[layer_i+1].weights_copy.length; j++) {
            if (this.layers[layer_i].dropped_out_i.includes(j)) continue;
            else this.layers[layer_i+1].weights_copy[j] = this.layers[layer_i+1].weights.shift();  
        }

        this.layers[layer_i].weights = this.layers[layer_i].weights_copy;
        this.layers[layer_i].layer = this.layers[layer_i].layer_copy;
        this.layers[layer_i+1].weights = this.layers[layer_i+1].weights_copy;
        this.dropped_out_i = [];
        //console.log(this.layers[0]);
    }
    
    sigmoid(z) {
        var bottom = math.add(1, math.exp(math.multiply(-1, z)));
        return math.dotDivide(1, bottom);
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
            else if (i==0) {
                this.layers[i].error = math.multiply(this.layers[i+1].delta, math.transpose(this.layers[i+1].weights));
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.layers[i].layer.map(v => this.sigmoid_d(v))) ;
            }
            else if (i<this.layers.length-1) {
                this.layers[i].error = math.multiply(this.layers[i+1].delta, math.transpose(this.layers[i+1].weights));
                this.layers[i].delta = math.dotMultiply(this.layers[i].error, this.layers[i].layer.map(v => this.sigmoid_d(v))) ;
            }
        }
        //update the weights
        for (var i=this.layers.length-1; i>=1; i--) {
            this.layers[i].weights = math.add(this.layers[i].weights, math.multiply(math.transpose(this.layers[i-1].layer), this.layers[i].delta));
        }
        this.layers[0].weights = math.add(this.layers[0].weights, math.multiply(math.transpose(this.input), this.layers[0].delta));
        //console.log(this.layers)
    }

    train(epochs, X, y) {
        for (var i=0; i<epochs; i++) {
            for (var x=0; x<X.length; x++) {
            this.input = [X[x]];
            this.Y = [y[x]];
            //this.dropout(0, 0.05);
            this.feedforward();
            this.backprop();
            console.log(this.output);
            //this.dropout_restore(0);
            }
        }
    }
}