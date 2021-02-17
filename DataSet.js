import * as math from 'mathjs';
import LineByLine from 'n-readlines';

export class DataSet {

    /**
     * https://stackoverflow.com/a/6274381/2032235
     * Shuffles array in place.
     * @param {Array} a An array containing the items.
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
     * @param {Array} labels_array An array containing the categorical values.
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

    /**
     * Maps a one-hot set to categorical value representation.
     * @param {Array} onehot_arrays An array containing the one-hot representation of categorical values.
     * @param {Object} label_arrays where the mapping will be assigned; provide an empty object initially.
     */
    static oneHotToLabels(onehot_arrays, label_arrays) {
        let curr_key = "";
        //let label_arrays = {};
        Object.keys(onehot_arrays).forEach( (label) => {
            let oh_arr = onehot_arrays[label];
            oh_arr.forEach( (elem) => {
                curr_key += elem;
            });
            label_arrays[curr_key] = label;
            curr_key = "";
        });
        return label_arrays;
    }

    /**
     * Imports a data set from a file having each case in a line and features separated by the specified separator.
     * @param {String} path_to_file  The input file path.
     * @param {String} separator The delimiter (default: comma).
     */
    static import(path_to_file, separator) {
        separator = separator || ',';
        const liner = new LineByLine(path_to_file);
        let line;
        let row = [], data = [];
        while(line = liner.next()) {
            row = line.toString().replace('\r', '').split(separator);
            data.push(row);
        }
        return data;
    }

    /**
     * Separates X from Y features from a data set and returns them in an object.
     * @param {Array} data The imported data set.
     * @param {Integer} Y_index The index of the output feature (default: the index of the last feature element of a row).
     * @param {Boolean} one_hot Whether to provide a one-hot representation of the output feature as well.
     */
    static separateXY(data, Y_index, one_hot, onehot_to_labels) {
        if (!data || !data.length) return { X: [], Y: [] };
        Y_index = Number.isNaN(Y_index) ? (data[0].length - 1) : Y_index;
        let X = [], Y = [], labels = [];
        data.forEach((row) => {
            let X_row;
            if (Y_index === 0) {
                X_row = row.slice(1, row.length);
            }else if (Y_index === row.length - 1) {
                X_row = row.slice(0, Y_index);
            }else {
                X_row = row.slice(0, Y_index);
                X_row = X_row.concat(row.slice(Y_index+1, row.length));
            }
            X_row.forEach( (elem, i, arr) => { arr[i] = parseFloat(elem); });
            X.push(X_row);
            Y.push(row[Y_index]);
            labels[row[Y_index]] = 1;
        });
        if (one_hot) {
            let labelsToOneHot = DataSet.labelsToOneHot(Object.keys(labels));
            DataSet.oneHotToLabels(labelsToOneHot, onehot_to_labels);
            let Y_one_hot = [];
            Y.forEach((label) => {
                Y_one_hot.push(labelsToOneHot[label]);
            });
            return { X: X, Y: Y, Y_one_hot: Y_one_hot};
        }
        return { X: X, Y: Y};
    }

    /**
     * Normalizes the provided data set.
     * @param {Array} data The data set to normalize.
     */
    static normalize(data) {
        let maxes = math.max(data, 0);
        data.forEach((elem, i, arr) => {
            arr[i] = math.dotDivide(arr[i], maxes);
            //replace NaN values with 0
            arr[i].forEach((elem, i , arr) => {
                arr[i] = Number.isNaN(elem) ? 0 : elem;
            });
        });
    }

    /**
     * Separates the resulting dataset object to a train set and a test set.
     * @param {Object} data The dataset object with X, Y and optionally Y_onehot property.
     * @param {Float} test_p The approximate portion [0 - 1] of the data that will be kept as a test set.
     */
    static separateTrainTest(data, test_p) {
        if (!data || !data.hasOwnProperty('X') || !data.hasOwnProperty('Y')) return null;
        let test_indexes = [];
        let cardinality = Math.floor(test_p*data.X.length);
        data.X.forEach((elem, i) => {
            if (Math.random() <= test_p) test_indexes.push(i);
        });
        if (test_indexes.length - cardinality > 0) { test_indexes.splice(0, test_indexes.length - cardinality); }
        let i=0;
        while (test_indexes.length < cardinality) {
            if (!test_indexes.includes(i)) test_indexes.push(i);
            i++;
        }
        let tt = { train: { X: [], Y: [] }, test: { X: [], Y: [] } };
        if (data.Y_one_hot) { tt.train["Y_one_hot"] = []; tt.test["Y_one_hot"] = []; }
        data.X.forEach((elem, i) => {
            if (test_indexes.includes(i)) {
                tt.test.X.push(data.X[i]);
                tt.test.Y.push(data.Y[i]);
                if (data.Y_one_hot) { tt.test.Y_one_hot.push(data.Y_one_hot[i]); }
            }else {
                tt.train.X.push(data.X[i]);
                tt.train.Y.push(data.Y[i]);
                if (data.Y_one_hot) { tt.train.Y_one_hot.push(data.Y_one_hot[i]); }
            }
        });
        return tt;
    }

}