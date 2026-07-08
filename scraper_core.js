// scraper_core.js
// Shared scraping logic used by both scrape_dom_tree.js (single URL) and
// batch_scrape.js (many URLs from a file).

const path = require('path');
const fs = require('fs');

const ATOMIC_TAGS = ['SVG', 'IMG', 'VIDEO', 'CANVAS', 'PICTURE', 'IFRAME'];
const SEMANTIC_TAGS = new Set(['nav', 'header', 'footer', 'aside', 'main', 'dialog', 'button',
  'a', 'img', 'svg', 'input', 'textarea', 'select', 'form', 'ul', 'li', 'table', 'video']);

function sameBbox(a, b) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

function prune(node, depth, maxDepth) {
  if (!node) return null;
  const cappedChildren = (maxDepth != null && depth >= maxDepth)
    ? []
    : node.children.map((c) => prune(c, depth + 1, maxDepth)).filter(Boolean);
  node.children = cappedChildren;

  const isSemantic = SEMANTIC_TAGS.has(node.tag) || node.role;

  if (node.children.length === 1 && sameBbox(node.bbox, node.children[0].bbox) && !node.text) {
    return node.children[0];
  }
  if (!isSemantic && node.children.length === 1 && !node.text) {
    return node.children[0];
  }
  if (node.children.length === 0 && !node.text && !isSemantic) {
    return null;
  }
  return node;
}

function countNodes(node) {
  if (!node) return 0;
  return 1 + node.children.reduce((sum, c) => sum + countNodes(c), 0);
}

// Scrapes a single URL using an already-open browser. Returns
// { ok: true, nodeCount, outDir } or { ok: false, error }.
async function scrapeUrl(browser, url, outRoot, maxDepth, timeoutMs) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  try {
    // 'domcontentloaded' fires as soon as the HTML is parsed -- much faster
    // and more reliable than 'networkidle', which never fires on sites with
    // continuous background traffic (analytics, ads, chat widgets, polling).
    // We don't fail the whole scrape even if this times out -- proceed with
    // whatever rendered anyway, since a partial page is still useful data.
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs || 25000 });
    } catch (navErr) {
      // swallow and continue -- page may still have useful content
    }
    // Give client-rendered apps (React/Vue SPAs) a moment to hydrate/paint
    // after DOMContentLoaded, since that event fires before JS has run.
    await page.waitForTimeout(2500);

    const tree = await page.evaluate((atomicTagsList) => {
      const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'META', 'LINK', 'TEMPLATE', 'BR']);
      const ATOMIC = new Set(atomicTagsList);
      const MIN_AREA = 16;

      function isVisible(rect, style) {
        if (rect.width * rect.height < MIN_AREA) return false;
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) === 0) return false;
        if (rect.left < 0 || rect.top < 0) return false;
        if (rect.right > window.innerWidth || rect.bottom > window.innerHeight) return false;
        return true;
      }

      // Many modern sites (YouTube, Reddit, Notion, ...) render real content
      // inside open Shadow DOM roots, which normal .children traversal can't
      // see. Pierce into el.shadowRoot when present so we don't lose that
      // content. Closed shadow roots (rare) still can't be accessed -- that's
      // a genuine limitation, not something fixable from page.evaluate.
      function getChildren(el) {
        const kids = Array.from(el.children);
        if (el.shadowRoot) {
          kids.push(...Array.from(el.shadowRoot.children));
        }
        return kids;
      }

      function walk(el) {
        if (!el || SKIP.has(el.tagName)) return null;
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        if (!isVisible(rect, style)) return null;

        let ownText = '';
        for (const node of el.childNodes) {
          if (node.nodeType === Node.TEXT_NODE) ownText += node.textContent;
        }
        ownText = ownText.trim();

        const bbox = [
          Math.round(rect.left + window.scrollX),
          Math.round(rect.top + window.scrollY),
          Math.round(rect.width),
          Math.round(rect.height),
        ];

        const children = [];
        if (!ATOMIC.has(el.tagName)) {
          for (const child of getChildren(el)) {
            const childNode = walk(child);
            if (childNode) children.push(childNode);
          }
        }

        return {
          tag: el.tagName.toLowerCase(),
          role: el.getAttribute('role') || null,
          id: el.id || null,
          classes: el.className && typeof el.className === 'string'
            ? el.className.split(' ').filter(Boolean)
            : [],
          bbox,
          text: ownText.slice(0, 200) || null,
          children,
        };
      }

      return walk(document.body);
    }, ATOMIC_TAGS);

    const pruned = prune(tree, 0, maxDepth);
    const nodeCount = countNodes(pruned);

    if (nodeCount === 0) {
      await page.close();
      return { ok: false, error: 'Empty tree after pruning -- likely shadow DOM, bot wall, or blank render. Check a manual screenshot.' };
    }

    const host = new URL(url).hostname;
    // Use hostname + short hash of full URL so different pages on the same
    // domain don't collide into one folder.
    const hash = Buffer.from(url).toString('base64').replace(/[/+=]/g, '').slice(0, 8);
    const outDir = path.join(outRoot, `${host}__${hash}`);
    fs.mkdirSync(outDir, { recursive: true });

    await page.screenshot({ path: path.join(outDir, 'screenshot.png') });
    fs.writeFileSync(path.join(outDir, 'tree.json'), JSON.stringify(pruned));
    fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify({
      url, scrapedAt: new Date().toISOString(), nodeCount, viewport: [1440, 900],
    }, null, 2));

    await page.close();
    return { ok: true, nodeCount, outDir };
  } catch (err) {
    await page.close().catch(() => { });
    return { ok: false, error: err.message };
  }
}

module.exports = { scrapeUrl };
