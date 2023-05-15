import * as math from 'mathjs';
import LineByLine from 'n-readlines';
import fs from 'fs';

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
     * Finds the mode of an array.
     * @param {Array} arr the input array.
     * @returns the mode element of the input array.
     */
    static mode(arr) {
        return arr.sort((a,b) =>
            arr.filter(v => v===a).length
          - arr.filter(v => v===b).length
        ).pop();
    };

    /**
     * Maps a categorical set to binary representation.
     * @param {Array} labels_array An array containing the categorical values.
     * @return {Object} mapping from categorical values to their respective binary array representation.
     */
    static labelsToBinary(labels_array) {
        let numLabelBits = labels_array.length.toString(2).length;
        let label_to_bin_map = {};
        let idx_bin_arr;
        let idx_bin;
        let idx_num_bits;
        labels_array.forEach((elem, idx, arr) => {
            idx_bin_arr = [];
            idx_bin = parseInt(idx).toString(2);
            idx_num_bits = idx_bin.length;
            for (let i = idx_num_bits; i < numLabelBits; i++) {
                idx_bin_arr.push(0);
            }
            idx_bin.split('').forEach(elem => idx_bin_arr.push(parseInt(elem)) );
            label_to_bin_map[elem] = idx_bin_arr;
            });
        return label_to_bin_map;
    }

    /**
     * Maps a categorical set to one-hot representation.
     * @param {Array} labels_array An array containing the categorical values.
     * @return {Object} mapping from categorical values to their respective one-hot array representation.
     */
    static labelsToOneHot(labels_array) {
        let label_to_onehot_map = {};
        let onehot_array;
        labels_array.forEach((elem, i, arr) => {
            onehot_array = math.zeros(arr.length)._data;
            onehot_array[arr.length -1 - i] = 1;
            label_to_onehot_map[elem] = onehot_array;
        });
        return label_to_onehot_map;
    }

    /**
     * Maps a one-hot set to categorical value representation.
     * @param {Array} label_to_bin_map An array containing the one-hot representation of categorical values.
     * @param {Object} bin_to_label_map where the mapping will be assigned; provide an empty object initially. Example mapping: { '10': 's3', '01': 's2', '00': 's1' }
     */
    static binaryToLabels(label_to_bin_map, bin_to_label_map) {
        Object.keys(label_to_bin_map).forEach( (label) => {
            bin_to_label_map[label_to_bin_map[label].join('')] = label;
        });
        return bin_to_label_map;
    }

    /**
     * Maps a one-hot set to categorical value representation.
     * @param {Array} label_to_onehot_map An array containing the one-hot representation of categorical values.
     * @param {Object} onehot_to_label_map where the mapping will be assigned; provide an empty object initially. Example mapping: { '100': 's3', '010': 's2', '001': 's1' }
     */
    static oneHotToLabels(label_to_onehot_map, onehot_to_label_map) {
        let curr_key = "";
        Object.keys(label_to_onehot_map).forEach( (label) => {
            let oh_arr = label_to_onehot_map[label];
            oh_arr.forEach( (elem) => {
                curr_key += elem;
            });
            onehot_to_label_map[curr_key] = label;
            curr_key = "";
        });
        return onehot_to_label_map;
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
     * @param {String} output_encoding 'ONEHOT', 'BINARY' or 'NONE representation of the output feature (default: 'NONE').
     */
    static separateXY(data, Y_index, output_encoding='NONE', encoding_to_label_map) {
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
        if (output_encoding=='ONEHOT') {
            let labelsToOneHot = DataSet.labelsToOneHot(Object.keys(labels));
            DataSet.oneHotToLabels(labelsToOneHot, encoding_to_label_map);
            let Y_one_hot = [];
            Y.forEach((label) => {
                Y_one_hot.push(labelsToOneHot[label]);
            });
            return { X: X, Y: Y, Y_one_hot: Y_one_hot};

        } else if (output_encoding=='BINARY') {
            let labelsToBinary = DataSet.labelsToBinary(Object.keys(labels));
            DataSet.binaryToLabels(labelsToBinary, encoding_to_label_map);
            let Y_bin = [];
            Y.forEach((label) => {
                Y_bin.push(labelsToBinary[label]);
            });
            return { X: X, Y: Y, Y_bin: Y_bin};
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
        else if (data.Y_bin) { tt.train["Y_bin"] = []; tt.test["Y_bin"] = []; }
        data.X.forEach((elem, i) => {
            if (test_indexes.includes(i)) {
                tt.test.X.push(data.X[i]);
                tt.test.Y.push(data.Y[i]);
                if (data.Y_one_hot) { tt.test.Y_one_hot.push(data.Y_one_hot[i]); }
                else if (data.Y_bin) { tt.test.bin.push(data.Y_bin[i]); }
            }else {
                tt.train.X.push(data.X[i]);
                tt.train.Y.push(data.Y[i]);
                if (data.Y_one_hot) { tt.train.Y_one_hot.push(data.Y_one_hot[i]); }
                else if (data.Y_bin) { tt.train.Y_bin.push(data.Y_bin[i]); }
            }
        });
        return tt;
    }

}