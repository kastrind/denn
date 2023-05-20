import * as math from 'mathjs';

export class Activation {

    /**
     * Sigmoid activation.
     * @param {Array} z  Input matrix.
     * @param {Boolean} derivative  True for sigmoid derivative
     */
    static sigmoid(z, derivative) {
        if (derivative === true) {
            return math.dotMultiply(z, math.subtract(1, z));
        }
        //var bottom = math.add(1, math.exp(math.multiply(-1, z)));
        var bottom = math.add(1, math.map(math.multiply(-1, z), math.exp));
        //console.log("z is: " + z[0]);
        //console.log("bottom is:" + bottom[0]);
        if (isNaN(bottom[0])) {
            //console.log("NAN!");
            //console.log("!!!!!!!!!!!!!!!!!!"+math.exp(math.multiply(-1, z)));
        }else {
            //console.log("NOT NAN!");
            //console.log("!!!!!!!!!!!!!!!!!!"+math.exp(math.multiply(-1, z)));
        }
        //console.log("activation is="+math.dotDivide(1, bottom));
        return math.dotDivide(1, bottom);
    }

    /**
     * Relu activation.
     * @param {Array} z  Input matrix.
     * @param {Boolean} derivative  True for relu derivative
     */
    static relu(z, derivative) {
        let return_z = [];
        if (derivative === true) {
            z.forEach(function (val, i) {
                if (val > 0) {
                    return_z[i] = math.random(0.99, 1);
                    //console.log(return_z[i]);
                }else {
                    return_z[i] = math.random(0, 0.01);
                    //console.log(return_z[i]);
                }
            });
            return return_z;
        }
        //econsole.log(z);
        let res = math.compare(0, z);
        res.forEach(function (val, i) {
            if (val == -1) {
                return_z[i] = z[i] * math.random(0.99, 1);
            }else {
                return_z[i] = math.random(0, 0.01) * z[i];
            }
        });
        return return_z;
    }

    static softPlus(z, derivative) {
        if (derivative === true) {
            return Activation.sigmoid(z, false);
        }else {
            // let return_z = [];
            // z.forEach(function (val, i) {
            //     return_z[i] = Math.min(0.99, Math.log(1 + Math.exp(val)));
            // });
            // return return_z;
            return math.map(math.log(math.add(1, math.map(z, math.exp))), function(value) { return math.min(math.random(0.99, 1), value); });
        }
    }

}