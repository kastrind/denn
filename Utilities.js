import * as math from 'mathjs';

export class Utilities {

    /**
     * https://stackoverflow.com/a/6274381/2032235
     * Shuffles array in place.
     * @param {Array} a items An array containing the items.
     */
    static shuffle(a) {
        var j, x, i;
        for (i = a.length - 1; i > 0; i--) {
            j = Math.floor(Math.random() * (i + 1));
            x = a[i];
            a[i] = a[j];
            a[j] = x;
        }
        return a;
    }

    /**
     * Maps a categorical set to one-hot representation.
     * @param {Array} labels_array items An array containing the categorical values.
     */
    static labelsToOneHot(labels_array) {
        let onehot_arrays = {};
        labels_array.forEach((elem, i, arr) => {
            let onehot_array = math.zeros(arr.length)._data;
            onehot_array[arr.length -1 - i] = 1;
            onehot_arrays[elem] = onehot_array;
        });
        return onehot_arrays;
    }

}