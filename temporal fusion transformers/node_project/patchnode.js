// Polyfill for Node.js 24 compatibility with tfjs-node
// util.isNullOrUndefined was removed in Node 24
const util = require('util');
if (!util.isNullOrUndefined) {
  util.isNullOrUndefined = (v) => v === null || v === undefined;
}
