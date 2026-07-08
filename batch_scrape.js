// batch_scrape.js
// Usage: node batch_scrape.js urls.txt [maxDepth]
//
// Reads a plain-text file (one URL per line, blank lines ignored),
// scrapes each one sequentially, and writes a run_log.json summary
// so you know which sites succeeded/failed and why.

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { scrapeUrl } = require('./scraper_core');

const urlFile = process.argv[2];
const maxDepth = process.argv[3] ? parseInt(process.argv[3], 10) : null;

if (!urlFile) {
  console.error('Usage: node batch_scrape.js urls.txt [maxDepth]');
  process.exit(1);
}

const urls = fs.readFileSync(urlFile, 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l && !l.startsWith('#'));

const OUT_ROOT = 'out';
const RETRIES = 1; // retry once on failure before giving up

async function main() {
  const browser = await chromium.launch();
  const results = [];

  for (let i = 0; i < urls.length; i++) {
    const url = urls[i];
    console.log(`[${i + 1}/${urls.length}] ${url}`);

    let attempt = 0;
    let result;
    while (attempt <= RETRIES) {
      result = await scrapeUrl(browser, url, OUT_ROOT, maxDepth, 25000);
      if (result.ok) break;
      attempt++;
      if (attempt <= RETRIES) console.log(`  retry ${attempt}...`);
    }

    if (result.ok) {
      console.log(`  ok -- ${result.nodeCount} nodes -> ${result.outDir}`);
    } else {
      console.log(`  FAILED -- ${result.error}`);
    }
    results.push({ url, ...result });
  }

  await browser.close();

  const succeeded = results.filter((r) => r.ok).length;
  const failed = results.filter((r) => !r.ok);

  fs.writeFileSync(path.join(OUT_ROOT, 'run_log.json'), JSON.stringify(results, null, 2));

  console.log('\n=== DONE ===');
  console.log(`${succeeded}/${urls.length} succeeded`);
  if (failed.length) {
    console.log('Failed URLs:');
    failed.forEach((r) => console.log(`  ${r.url} -- ${r.error}`));
  }
  console.log('Full log written to out/run_log.json');
}

main();
