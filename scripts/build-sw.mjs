// Stamps the service worker template with this build's id and the OCR runtime version,
// and writes public/sw.js (gitignored). Runs before every `dev` and `build`.
//
//  - BUILD_ID changes every deploy, so the cached HTML shells are replaced with each release.
//  - OCR_VERSION changes only when tesseract.js / its core / the language data change, so
//    the ~7 MB OCR files stay cached across ordinary deploys.

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");

const version = (name) => JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve(`${name}/package.json`)), "package.json"), "utf8")).version;
const ocrVersion = crypto
  .createHash("sha1")
  .update(["tesseract.js", "tesseract.js-core", "@tesseract.js-data/eng"].map((n) => `${n}@${version(n)}`).join("|"))
  .digest("hex")
  .slice(0, 10);

const buildId = (process.env.VERCEL_GIT_COMMIT_SHA || String(Date.now())).slice(0, 12);

const template = fs.readFileSync(path.join(root, "scripts", "sw.template.js"), "utf8");
if (!template.includes("__BUILD_ID__") || !template.includes("__OCR_VERSION__")) {
  throw new Error("build-sw: template placeholders missing");
}
fs.writeFileSync(
  path.join(root, "public", "sw.js"),
  template.replaceAll("__BUILD_ID__", buildId).replaceAll("__OCR_VERSION__", ocrVersion),
);
console.log(`build-sw: public/sw.js (build ${buildId}, ocr ${ocrVersion})`);
