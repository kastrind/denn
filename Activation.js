import * as math from 'mathjs';

export class Activation {

    /**
     * Sigmoid activation.
     * @param {Array} z  Input matrix.
     * @param {Boolean} derivative  True for sigmoid derivative.
     */
    static sigmoid(z, derivative) {
        if (derivative === true) {
            return math.dotMultiply(z, math.subtract(1, z));
        }
        var bottom = math.add(1, math.map(math.multiply(-1, z), math.exp));
        return math.dotDivide(1, bottom);
    }

    /**
     * Relu activation.
     * @param {Array} z  Input matrix.
     * @param {Boolean} derivative  True for relu derivative.
     */
    static relu(z, derivative) {
        let return_z = [];
        if (derivative === true) {
            z.forEach(function (val, i) {
                if (val > 0) {
                    return_z[i] = math.random(0.99, 1);
                }else {
                    return_z[i] = math.random(0, 0.01);
                }
            });
            return return_z;
        }
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

    /**
     * Softplus activation.
     * @param {Array} z  Input matrix.
     * @param {Boolean} derivative  True for softplus derivative.
     * @returns 
     */
    static softPlus(z, derivative) {
        if (derivative === true) {
            return Activation.sigmoid(z, false);
        }else {
            return math.map(math.log(math.add(1, math.map(z, math.exp))), function(value) { return math.min(math.random(0.99, 1), value); });
        }
    }

}