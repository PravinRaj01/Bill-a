// Turns the two Kaggle receipt datasets in bench/kaggle_datasets/ into the layout
// the OCR benchmark reads:
//
//   bench/receipts/kaggle-ocr-<id>.jpg       labelled images  (dataset 1)
//   bench/receipts/truth.kaggle-ocr.csv       file,grand_total,item_lines,notes
//   bench/receipts/truth.kaggle-ocr.json      full annotation text per image
//   bench/receipts-unlabeled/pdf-*.jpg        rendered PDF scans (dataset 2), no ground truth
//   bench/receipts-unlabeled/manifest.csv     category / country / year / source for each
//
// Run:  node bench/prepare-datasets.mjs [--pdf-sample=40] [--seed=1]
// Everything it writes is gitignored (see .gitignore); re-running overwrites it.

import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const SRC = path.join(ROOT, "bench", "kaggle_datasets");
const D1 = path.join(SRC, "OCR Receipts text detection dataset");
const D2 = path.join(SRC, "my receipts pdf scans dataset");
const LABELLED = path.join(ROOT, "bench", "receipts");
const UNLABELLED = path.join(ROOT, "bench", "receipts-unlabeled");

const arg = (name, dflt) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.split("=")[1] : dflt;
};
const PDF_SAMPLE = Number(arg("pdf-sample", 40));
const SEED = Number(arg("seed", 1));

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
const unxml = (s) =>
  s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

// ---------------------------------------------------------------------------
// Dataset 1: CVAT annotations (shop / item / total / date_time boxes with text)
// ---------------------------------------------------------------------------

/** "TOTAL $38.68" -> 38.68 ; "TOTAL 1,234.50" -> 1234.5 ; null if no amount. */
function parseAmount(text) {
  const m = [...text.replace(/(\d)[ ](\d{3})\b/g, "$1$2").matchAll(/(\d{1,3}(?:[,]\d{3})*|\d+)[.,](\d{2})\b/g)];
  if (m.length === 0) return null;
  const last = m[m.length - 1];
  return Number(`${last[1].replace(/,/g, "")}.${last[2]}`);
}

function prepareDataset1() {
  const xml = fs.readFileSync(path.join(D1, "annotations.xml"), "utf8");
  const images = [...xml.matchAll(/<image id="(\d+)" name="([^"]+)" width="(\d+)" height="(\d+)">([\s\S]*?)<\/image>/g)];
  fs.mkdirSync(LABELLED, { recursive: true });

  const rows = [];
  const detail = [];
  for (const [, id, name, width, height, body] of images) {
    const boxes = [...body.matchAll(/<box label="([a-z_]+)"[^>]*>\s*<attribute name="text">([\s\S]*?)<\/attribute>/g)].map(
      (b) => ({ label: b[1], text: unxml(b[2]).trim() }),
    );
    const totals = boxes.filter((b) => b.label === "total");
    const items = boxes.filter((b) => b.label === "item");
    const shop = boxes.find((b) => b.label === "shop")?.text ?? "";
    const date = boxes.find((b) => b.label === "date_time")?.text ?? "";

    const ext = path.extname(name).toLowerCase();
    const outName = `kaggle-ocr-${id.padStart(2, "0")}${ext}`;
    fs.copyFileSync(path.join(D1, name), path.join(LABELLED, outName));

    const parsedTotals = totals.map((t) => parseAmount(t.text)).filter((v) => v !== null);
    const notes = [];
    if (totals.length === 0) notes.push("no total annotated");
    else if (parsedTotals.length === 0) notes.push(`total text unparseable: ${totals[0].text}`);
    if (parsedTotals.length > 1) notes.push(`${parsedTotals.length} total boxes, using the last`);
    if (items.length === 0) notes.push("no item lines annotated");

    const grandTotal = parsedTotals.length ? parsedTotals[parsedTotals.length - 1] : "";
    rows.push([outName, grandTotal === "" ? "" : grandTotal.toFixed(2), items.length, `${shop}${notes.length ? " | " + notes.join("; ") : ""}`]);
    detail.push({
      file: outName,
      width: Number(width),
      height: Number(height),
      shop,
      date,
      grandTotal: grandTotal === "" ? null : grandTotal,
      totalText: totals.map((t) => t.text),
      items: items.map((i) => ({ text: i.text, amount: parseAmount(i.text) })),
    });
  }

  fs.writeFileSync(
    path.join(LABELLED, "truth.kaggle-ocr.csv"),
    ["file,grand_total,item_lines,notes", ...rows.map((r) => r.map(csvCell).join(","))].join("\n") + "\n",
  );
  fs.writeFileSync(path.join(LABELLED, "truth.kaggle-ocr.json"), JSON.stringify(detail, null, 2));
  return { images: images.length, withTotal: rows.filter((r) => r[1] !== "").length, rows };
}

// ---------------------------------------------------------------------------
// Dataset 2: PDF scans -> stratified sample of JPEGs (no ground truth)
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function* walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(p);
    else if (/\.pdf$/i.test(e.name)) yield p;
  }
}

async function prepareDataset2() {
  const mupdf = await import("mupdf");
  // Bill-splitting is about restaurants/cafes and shops, so sample those; the
  // dataset is also full of train tickets and hotel folios that would just add noise.
  const WANT = { restaurant: 0.55, cafe: 0.25, retail: 0.2 };

  const byCat = {};
  for (const file of walk(D2)) {
    const parts = path.relative(D2, file).split(path.sep); // year/country/category/file
    if (parts.length !== 4) continue;
    const [year, country, category] = parts;
    if (!(category in WANT)) continue;
    (byCat[category] ??= []).push({ file, year, country, category });
  }

  const rand = mulberry32(SEED);
  const chosen = [];
  for (const [cat, share] of Object.entries(WANT)) {
    const pool = byCat[cat] ?? [];
    const shuffled = pool.map((v) => [rand(), v]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    chosen.push(...shuffled.slice(0, Math.round(PDF_SAMPLE * share)));
  }

  fs.rmSync(UNLABELLED, { recursive: true, force: true });
  fs.mkdirSync(UNLABELLED, { recursive: true });

  const manifest = ["file,category,country,year,pages,width,height,source"];
  let n = 0;
  const failures = [];
  for (const c of chosen) {
    try {
      const doc = mupdf.Document.openDocument(fs.readFileSync(c.file), "application/pdf");
      const pages = doc.countPages();
      const page = doc.loadPage(0); // first page only: receipts are one page, the rest is usually a back side
      const [x0, y0, x1, y1] = page.getBounds();
      // Aim for ~2000 px on the long edge — about what a phone photo gives the app before its own downscale.
      const scale = 2000 / Math.max(x1 - x0, y1 - y0);
      const pix = page.toPixmap(mupdf.Matrix.scale(scale, scale), mupdf.ColorSpace.DeviceRGB, false, true);
      const out = `pdf-${c.category}-${String(++n).padStart(3, "0")}.jpg`;
      fs.writeFileSync(path.join(UNLABELLED, out), pix.asJPEG(85, false));
      manifest.push([out, c.category, c.country, c.year, pages, pix.getWidth(), pix.getHeight(), path.relative(SRC, c.file)].map(csvCell).join(","));
    } catch (e) {
      failures.push(`${path.relative(SRC, c.file)}: ${e.message}`);
    }
  }
  fs.writeFileSync(path.join(UNLABELLED, "manifest.csv"), manifest.join("\n") + "\n");
  return { available: Object.fromEntries(Object.entries(byCat).map(([k, v]) => [k, v.length])), written: n, failures };
}

// ---------------------------------------------------------------------------

const d1 = prepareDataset1();
console.log(`dataset 1: ${d1.images} images -> bench/receipts/, ${d1.withTotal} with a parsed grand total`);
const d2 = await prepareDataset2();
console.log(`dataset 2: available ${JSON.stringify(d2.available)}; rendered ${d2.written} -> bench/receipts-unlabeled/`);
if (d2.failures.length) console.log(`  ${d2.failures.length} failed:\n  ${d2.failures.join("\n  ")}`);
