import * as math from 'mathjs';

export class Activation {

    static sigmoid(z, derivative) {
        if (derivative === true) {
            return math.dotMultiply(z, math.subtract(1, z));
        }
        var bottom = math.add(1, math.exp(math.multiply(-1, z)));
        return math.dotDivide(1, bottom);
    }

}