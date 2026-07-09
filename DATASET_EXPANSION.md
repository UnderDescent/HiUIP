# Expanding the Dataset

The repo has the needed pieces:

- `batch_scrape.js` scrapes URL lists into screenshot/tree/meta folders.
- `split_dataset.py` creates train/eval site splits.
- `build_direct_children_dataset.py` builds parent-crop -> direct-child-box JSONL.
- `verify_direct_children_dataset.py` checks every crop and child box is inside the image.
- `visualize_direct_children_examples.py` renders random annotated examples.

Suggested flow:

```powershell
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe batch_scrape.js urls_expand.txt 8 data_import\out_expanded
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe split_dataset.py data_import\out_expanded 8
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe build_direct_children_dataset.py data_import\out_expanded train_sites.json data_import\train_direct_children_expanded.jsonl
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe build_direct_children_dataset.py data_import\out_expanded eval_sites.json data_import\eval_direct_children_expanded.jsonl
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe verify_direct_children_dataset.py data_import\train_direct_children_expanded.jsonl data_import\eval_direct_children_expanded.jsonl
C:\Users\Student3\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe visualize_direct_children_examples.py data_import\train_direct_children_expanded.jsonl data_import\direct_child_examples_expanded --count 48
```

Notes:

- Some sites will block automation or render login walls. That is expected.
- Prefer many distinct pages over many near-duplicates from one domain.
- After visual inspection, merge the expanded JSONL with the current JSONL only if the examples look clean.

## External datasets to check

### WebUI

WebUI is the closest match for this project if the files can be obtained. The paper describes about 400k rendered web pages, each with screenshots plus browser-derived accessibility-tree metadata and element layout boxes. That is almost the exact raw material needed to build our parent-crop -> direct-child-box rows.

The expected conversion path is:

1. Load each viewport screenshot.
2. Load the accessibility/tree metadata for that screenshot.
3. Use each visible node as a possible parent crop.
4. Use only its direct visible children as labels.
5. Apply the same filters we already use here: child boxes inside crop, no giant full-page crops, collapse one-child wrapper chains where useful.

Status: found. The project repo is `https://github.com/js0nwu/webui`, and the public Hugging Face releases are under `biglab/`.

Best first target:

- `biglab/webui-7kbal`: balanced higher-quality subset, 7000 sample ids in `balanced_7k.json`, data split into `balanced_7k.zip.001` and `balanced_7k.zip.002`.

Other useful releases:

- `biglab/webui-7k`: regular 7k subset, about 8.36 GB.
- `biglab/webui-70k` and `biglab/webui-350k`: bigger versions after the converter is proven.
- `biglab/webui-*-elements`: flattened element datasets in Parquet. Useful for element detection, but probably less useful for our direct-child hierarchy task unless they preserve parent-child relationships.

Next action: download `biglab/webui-7kbal`, extract it, inspect one sample's `*-axtree.json.gz` and `*-bb.json.gz`, then write a WebUI-to-direct-children converter.

### WebSRC

WebSRC is less perfect for this task, but it is publicly described as web pages with HTML, screenshots, and metadata. If WebUI is unavailable, WebSRC may be easier to obtain and could be converted with a similar tree+box pipeline if its metadata includes usable element bounds.

### RICO / MobileViews

These are mobile-app screenshot + hierarchy datasets. They are not web pages, but they can still teach the same visual idea: a crop of a parent UI region should return only direct children. Use these after WebUI/WebSRC, or if we want more non-web UI variety.
