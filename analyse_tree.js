// analyze_tree.js
// Usage: node analyze_tree.js out/<hostname>/tree.json

const fs = require('fs');
const path = process.argv[2];
if (!path) {
    console.error('Usage: node analyze_tree.js <path-to-tree.json>');
    process.exit(1);
}

const tree = JSON.parse(fs.readFileSync(path, 'utf8'));

let totalNodes = 0;
const tagCounts = {};
const depthCounts = {};
let maxDepth = 0;
let maxChildren = 0;
let maxChildrenNode = null;

// Track subtree "shape signatures" to find repeated structures (e.g. 50 identical list-item subtrees)
const shapeCounts = {};

function shapeSignature(node, depth) {
    if (depth > 3) return node.tag; // cap signature depth so it's still useful
    const childSigs = (node.children || []).map(c => shapeSignature(c, depth + 1));
    return `${node.tag}(${childSigs.join(',')})`;
}

function walk(node, depth) {
    totalNodes++;
    tagCounts[node.tag] = (tagCounts[node.tag] || 0) + 1;
    depthCounts[depth] = (depthCounts[depth] || 0) + 1;
    maxDepth = Math.max(maxDepth, depth);

    const nChildren = (node.children || []).length;
    if (nChildren > maxChildren) {
        maxChildren = nChildren;
        maxChildrenNode = { tag: node.tag, id: node.id, classes: node.classes, depth };
    }

    const sig = shapeSignature(node, 0);
    shapeCounts[sig] = (shapeCounts[sig] || 0) + 1;

    (node.children || []).forEach(c => walk(c, depth + 1));
}

walk(tree, 0);

console.log('=== SUMMARY ===');
console.log(`Total nodes: ${totalNodes}`);
console.log(`Max depth: ${maxDepth}`);
console.log(`Node with most direct children: ${maxChildren} children ->`, maxChildrenNode);

console.log('\n=== TAG COUNTS (top 20) ===');
Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .forEach(([tag, count]) => console.log(`${tag}: ${count}`));

console.log('\n=== NODES PER DEPTH LEVEL ===');
Object.entries(depthCounts)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .forEach(([d, count]) => console.log(`depth ${d}: ${count}`));

console.log('\n=== MOST REPEATED SUBTREE SHAPES (top 10) ===');
console.log('(high count = same structure repeated many times, e.g. list items/cards)');
Object.entries(shapeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .forEach(([sig, count]) => console.log(`x${count}: ${sig.slice(0, 120)}`));