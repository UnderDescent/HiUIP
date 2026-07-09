// scrape_dom_tree.js
// Usage: node scrape_dom_tree.js <url> [maxDepth] [outRoot]
//
// Setup (one-time):
//   npm init -y
//   npm install playwright
//   npx playwright install chromium
//
// Output (written under out/<hostname>__<hash>/):
//   screenshot.png   -- viewport screenshot
//   tree.json        -- pruned nested box tree aligned to the screenshot
//   meta.json        -- url, timestamp, node count

const { chromium } = require('playwright');
const { scrapeUrl } = require('./scraper_core');

const url = process.argv[2];
const maxDepth = process.argv[3] ? parseInt(process.argv[3], 10) : null;
const outRoot = process.argv[4] || 'out';

if (!url) {
    console.error('Usage: node scrape_dom_tree.js <url> [maxDepth]');
    process.exit(1);
}

async function main() {
    const browser = await chromium.launch();
    const result = await scrapeUrl(browser, url, outRoot, maxDepth, 25000);
    await browser.close();

    if (result.ok) {
        console.log(`Saved ${result.nodeCount} nodes to ${result.outDir}`);
    } else {
        console.error(`Failed: ${result.error}`);
        process.exit(1);
    }
}

main();
