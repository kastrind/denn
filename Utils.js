Array.prototype.toBinary = function(threshold) {
	for (let i=0; i < this.length; i++) {
	  this[i] = this[i] >= threshold ? 1 : 0;
	}
	return this;
};

export class Utils {

}
