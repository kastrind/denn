import * as math from 'mathjs';

export class Activation {

    static sigmoid(z, derivative) {
        if (derivative === true) {
            return math.dotMultiply(z, math.subtract(1, z));
        }
        var bottom = math.add(1, math.exp(math.multiply(-1, z)));
        return math.dotDivide(1, bottom);
    }

    static relu(z, derivative) {
        let return_z = [];
        if (derivative === true) {
            z.forEach(function (val, i) {
                if (val > 0) {
                    return_z[i] = math.random(0.9, 1);
                }else {
                    return_z[i] = math.random(0, 0.1);
                }
            });
            return return_z;
        }
        let res = math.compare(0, z);
        res.forEach(function (val, i) {
            if (val == -1) {
                return_z[i] = z[i];
            }else {
                return_z[i] = math.random(0, 0.1);
            }
        });
        return return_z;
    }

}