import { DataSet } from './DataSet';

console.log(DataSet.labelsToBinary(['s1', 's2', 's3']));

let label_arrays = {};
DataSet.binaryToLabels({ s1: [ 0, 0 ], s2: [ 0, 1 ], s3: [ 1, 0 ] }, label_arrays);
console.log(label_arrays);

let label_arrays_2 = {};
DataSet.oneHotToLabels({ s1: [ 0, 0, 1 ], s2: [ 0, 1, 0 ], s3: [ 1, 0, 0 ] }, label_arrays_2);
console.log(label_arrays_2);