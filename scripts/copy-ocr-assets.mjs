// Copies the tesseract.js runtime (worker, WASM core, English language data) from
// node_modules into public/ocr/ so the browser loads everything from OUR origin.
// Without this, tesseract.js fetches them from a public CDN at scan time — which
// would break offline scanning and the strict CSP planned for the API-key pages.
//
// Runs automatically before `dev` and `build` (see package.json). The output is
// gitignored; it is regenerated from the pinned npm packages every time.

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
const out = path.join(root, "public", "ocr");

const pkgDir = (name) => path.dirname(require.resolve(`${name}/package.json`));
const copy = (from, to) => {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
};

const tess = pkgDir("tesseract.js");
const core = pkgDir("tesseract.js-core");
const eng = pkgDir("@tesseract.js-data/eng");

fs.rmSync(out, { recursive: true, force: true });
copy(path.join(tess, "dist", "worker.min.js"), path.join(out, "worker.min.js"));

// Only the LSTM cores are used (tesseract.js runs the LSTM engine); the browser
// downloads just one of the three SIMD variants it supports.
let coreFiles = 0;
for (const f of fs.readdirSync(core)) {
  if (/^tesseract-core.*lstm.*\.(js|wasm)$/.test(f)) {
    copy(path.join(core, f), path.join(out, "core", f));
    coreFiles++;
  }
}
// "best_int": the same model the benchmark measured (~3 MB gzipped).
copy(path.join(eng, "4.0.0_best_int", "eng.traineddata.gz"), path.join(out, "lang", "eng.traineddata.gz"));

if (coreFiles === 0) throw new Error("copy-ocr-assets: no tesseract core files found — package layout changed?");
console.log(`copy-ocr-assets: worker + ${coreFiles} core files + eng.traineddata.gz -> public/ocr/`);
