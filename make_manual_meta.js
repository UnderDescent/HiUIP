// make_manual_meta.js
// Usage: node make_manual_meta.js <path-to-folder> <original-url>
// e.g.:  node make_manual_meta.js out/amazon.com__manual https://www.amazon.com

const fs = require('fs');
const path = require('path');

const folder = process.argv[2];
const url = process.argv[3];

if (!folder || !url) {
  console.error('Usage: node make_manual_meta.js <folder> <url>');
  process.exit(1);
}

const treePath = path.join(folder, 'tree.json');
const tree = JSON.parse(fs.readFileSync(treePath, 'utf8'));

function countNodes(node) {
  if (!node) return 0;
  return 1 + (node.children || []).reduce((s, c) => s + countNodes(c), 0);
}

const nodeCount = countNodes(tree);
const [x, y, width, height] = tree.bbox || [0, 0, null, null];

const meta = {
  url,
  scrapedAt: new Date().toISOString(),
  nodeCount,
  viewport: [width, height],
  method: 'manual', // flags this as a hand-captured entry, not from batch_scrape.js
};

fs.writeFileSync(path.join(folder, 'meta.json'), JSON.stringify(meta, null, 2));
console.log(`Wrote meta.json: ${nodeCount} nodes, viewport ${width}x${height}`);
