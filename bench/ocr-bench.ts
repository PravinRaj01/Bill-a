// OCR engine benchmark. Runs each engine config on every labelled and unlabelled
// receipt under bench/, through the SAME preprocessing (1600 px longest edge, JPEG
// q80 — what the app will send) and the SAME parser (lib/ocr/parse-lines.ts), then
// writes bench/results/ocr-bench.md (committed, aggregate only) and
// bench/results/ocr-bench.json (local detail).
//
//   npx tsx bench/ocr-bench.ts [--quick] [--only=tesseract,paddle-tiny] [--merge]
//
// --merge keeps the previous results of every config NOT re-run this time, so one
// engine can be re-benchmarked without redoing the slow ones.
//
// Node-side numbers: WASM-in-browser latency will differ (see the report's caveats);
// accuracy does not depend on where it runs.

import fs from "node:fs";
import path from "node:path";
import sharp from "sharp";
import { createPaddleEngine } from "../lib/ocr/engines/paddle";
import { createTesseractEngine } from "../lib/ocr/engines/tesseract";
import { extractAmounts, parseReceiptLines, type ParsedReceipt } from "../lib/ocr/parse-lines";
import type { OcrEngine, OcrOutput } from "../lib/ocr/types";

const ROOT = path.resolve(import.meta.dirname, "..");
const LABELLED = path.join(ROOT, "bench", "receipts");
const UNLABELLED = path.join(ROOT, "bench", "receipts-unlabeled");
const OUT = path.join(ROOT, "bench", "results");

const QUICK = process.argv.includes("--quick");
const MERGE = process.argv.includes("--merge");
const ONLY = process.argv.find((a) => a.startsWith("--only="))?.split("=")[1]?.split(",");

// ---------------------------------------------------------------------------

type Variant = "plain" | "gray";

async function prepare(file: string, variant: Variant) {
  let img = sharp(file).rotate().resize({ width: 1600, height: 1600, fit: "inside", withoutEnlargement: true });
  if (variant === "gray") img = img.greyscale().normalise();
  const { data, info } = await img.jpeg({ quality: 80 }).toBuffer({ resolveWithObject: true });
  const bytes = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
  return { bytes, width: info.width, height: info.height };
}

interface Truth {
  file: string;
  grandTotalCents: number | null;
  itemAmountsCents: number[];
}

function loadTruth(): Truth[] {
  const out = new Map<string, Truth>();
  const json = path.join(LABELLED, "truth.kaggle-ocr.json");
  if (fs.existsSync(json)) {
    for (const r of JSON.parse(fs.readFileSync(json, "utf8")) as {
      file: string;
      grandTotal: number | null;
      items: { amount: number | null }[];
    }[]) {
      out.set(r.file, {
        file: r.file,
        grandTotalCents: r.grandTotal === null ? null : Math.round(r.grandTotal * 100),
        itemAmountsCents: r.items.map((i) => i.amount).filter((a): a is number => a !== null).map((a) => Math.round(a * 100)),
      });
    }
  }
  // The user's own photos: truth.csv (file,grand_total,item_lines,notes) — totals only.
  const csv = path.join(LABELLED, "truth.csv");
  if (fs.existsSync(csv)) {
    for (const line of fs.readFileSync(csv, "utf8").split(/\r?\n/).slice(1)) {
      const [file, total] = line.split(",");
      if (file && total && fs.existsSync(path.join(LABELLED, file))) {
        out.set(file, { file, grandTotalCents: Math.round(Number(total) * 100), itemAmountsCents: [] });
      }
    }
  }
  return [...out.values()].filter((t) => fs.existsSync(path.join(LABELLED, t.file)));
}

// ---------------------------------------------------------------------------

interface Config {
  id: string;
  variant: Variant;
  make: () => OcrEngine;
}

// PaddleOCR (and the ~250 MB onnxruntime-node it needs) is NOT a project dependency: it
// lost the benchmark (see bench/results/ocr-bench.md) and would bloat every Vercel build.
// To re-run its rows:  npm i --no-save ppu-paddle-ocr onnxruntime-node
const paddle = (label: string, preset: "V6_TINY_MODEL" | "V6_SMALL_MODEL" | "V5_EN_MOBILE_MODEL") =>
  (): OcrEngine => {
    let inner: OcrEngine | null = null;
    return {
      name: label,
      async init() {
        const spec = "ppu-paddle-ocr"; // via a variable so the type-checker doesn't require the package
        const mod = await import(spec).catch(() => {
          throw new Error("PaddleOCR benchmarks need:  npm i --no-save ppu-paddle-ocr onnxruntime-node");
        });
        inner = createPaddleEngine(() => new mod.PaddleOcrService({ model: mod[preset] }), label);
        await inner.init();
      },
      recognize: (image, size) => inner!.recognize(image, size),
      dispose: async () => void (await inner?.dispose()),
    };
  };

const ENGINES: { id: string; make: () => OcrEngine }[] = [
  { id: "tesseract", make: () => createTesseractEngine() },
  { id: "paddle-tiny", make: paddle("paddle-tiny", "V6_TINY_MODEL") },
  { id: "paddle-small", make: paddle("paddle-small", "V6_SMALL_MODEL") },
  { id: "paddle-v5-en", make: paddle("paddle-v5-en", "V5_EN_MOBILE_MODEL") },
];

const CONFIGS: Config[] = ENGINES.filter((e) => (ONLY ? ONLY.includes(e.id) : !e.id.startsWith("paddle"))).flatMap((e) =>
  (["plain", "gray"] as Variant[]).map((variant) => ({ id: `${e.id}/${variant}`, variant, make: e.make })),
);

// ---------------------------------------------------------------------------

interface Run {
  config: string;
  image: string;
  set: "labelled" | "unlabelled";
  ms: number;
  error?: string;
  parsed?: ParsedReceipt;
  ocrChars?: number;
  // labelled only
  totalExact?: boolean;
  totalSeen?: boolean;
  itemsSeen?: number; // matched annotated item amounts found anywhere in the OCR text
  itemsParsed?: number; // matched annotated item amounts found among parsed item prices
  itemsAnnotated?: number;
}

/** How many of `wanted` cents appear in `pool` (each pool entry used once). */
function multisetHits(wanted: number[], pool: number[]): number {
  const left = [...pool];
  let hits = 0;
  for (const w of wanted) {
    const i = left.indexOf(w);
    if (i >= 0) {
      hits++;
      left.splice(i, 1);
    }
  }
  return hits;
}

/** Same reconciliation the parser applies: does the item sum agree with the receipt's own subtotal? */
const consistent = (p: ParsedReceipt) =>
  p.totalSource === "keyword" && p.items.length > 0 && !p.warnings.some((w) => w.startsWith("Items add up"));

const pct = (n: number, d: number) => (d ? `${((100 * n) / d).toFixed(0)}%` : "n/a");
const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : 0;
};

async function main() {
  const truth = loadTruth();
  let unlabelled = fs.existsSync(UNLABELLED)
    ? fs.readdirSync(UNLABELLED).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort()
    : [];
  let labelled = truth;
  if (QUICK) {
    labelled = labelled.slice(0, 5);
    unlabelled = unlabelled.slice(0, 5);
  }
  console.log(`${CONFIGS.length} configs × (${labelled.length} labelled + ${unlabelled.length} unlabelled)`);

  const prepared = new Map<string, Awaited<ReturnType<typeof prepare>>>();
  const prep = async (file: string, variant: Variant) => {
    const key = `${variant}:${file}`;
    if (!prepared.has(key)) prepared.set(key, await prepare(file, variant));
    return prepared.get(key)!;
  };

  const runs: Run[] = [];
  const initMs: Record<string, number> = {};

  if (MERGE && fs.existsSync(path.join(OUT, "ocr-bench.json"))) {
    const prev = JSON.parse(fs.readFileSync(path.join(OUT, "ocr-bench.json"), "utf8")) as {
      rows: { id: string; init: number }[];
      runs: Run[];
    };
    const rerun = new Set(CONFIGS.map((c) => c.id));
    const known = new Set(ENGINES.flatMap((e) => (["plain", "gray"] as const).map((v) => `${e.id}/${v}`)));
    for (const r of prev.runs) if (known.has(r.config) && !rerun.has(r.config)) runs.push(r);
    for (const r of prev.rows) if (known.has(r.id) && !rerun.has(r.id)) initMs[r.id] = r.init;
    console.log(`merged ${new Set(runs.map((r) => r.config)).size} configs from the previous run`);
  }

  for (const cfg of CONFIGS) {
    const engine = cfg.make();
    const t0 = performance.now();
    try {
      await engine.init();
    } catch (e) {
      console.log(`${cfg.id}: init failed: ${(e as Error).message}`);
      continue;
    }
    initMs[cfg.id] = performance.now() - t0;
    process.stdout.write(`${cfg.id} `);

    const doOne = async (file: string, set: Run["set"], t?: Truth) => {
      const run: Run = { config: cfg.id, image: path.basename(file), set, ms: 0 };
      try {
        const p = await prep(file, cfg.variant);
        const ocr: OcrOutput = await engine.recognize(p.bytes, { width: p.width, height: p.height });
        const parsed = parseReceiptLines(ocr);
        run.ms = ocr.ms;
        run.parsed = parsed;
        run.ocrChars = ocr.text.length;
        if (t) {
          const seen = extractAmounts(ocr.text);
          run.totalExact = t.grandTotalCents !== null && parsed.receipt.total === t.grandTotalCents;
          run.totalSeen = t.grandTotalCents !== null && seen.includes(t.grandTotalCents);
          run.itemsAnnotated = t.itemAmountsCents.length;
          run.itemsSeen = multisetHits(t.itemAmountsCents, seen);
          run.itemsParsed = multisetHits(t.itemAmountsCents, parsed.items.map((i) => i.totalPrice));
        }
      } catch (e) {
        run.error = (e as Error).message;
      }
      runs.push(run);
    };

    for (const t of labelled) await doOne(path.join(LABELLED, t.file), "labelled", t);
    for (const f of unlabelled) await doOne(path.join(UNLABELLED, f), "unlabelled");
    await engine.dispose();
    process.stdout.write("done\n");
  }

  // ------------------------------- report ---------------------------------
  const ids = [...new Set(runs.map((r) => r.config))];
  const rows = ids.map((id) => {
    const lab = runs.filter((r) => r.config === id && r.set === "labelled");
    const unl = runs.filter((r) => r.config === id && r.set === "unlabelled");
    const labWithTotal = lab.filter((r) => r.totalExact !== undefined && !r.error);
    const truthWithTotal = labelled.filter((t) => t.grandTotalCents !== null).length;
    const annotated = lab.reduce((s, r) => s + (r.itemsAnnotated ?? 0), 0);
    const ms = runs.filter((r) => r.config === id && !r.error).map((r) => r.ms);
    return {
      id,
      totalExact: lab.filter((r) => r.totalExact).length,
      totalSeen: lab.filter((r) => r.totalSeen).length,
      truthWithTotal,
      itemsSeen: lab.reduce((s, r) => s + (r.itemsSeen ?? 0), 0),
      itemsParsed: lab.reduce((s, r) => s + (r.itemsParsed ?? 0), 0),
      annotated,
      unlKeyword: unl.filter((r) => r.parsed?.totalSource === "keyword").length,
      unlConsistent: unl.filter((r) => r.parsed && consistent(r.parsed)).length,
      unlN: unl.length,
      errors: runs.filter((r) => r.config === id && r.error).length,
      p50: quantile(ms, 0.5),
      p95: quantile(ms, 0.95),
      init: initMs[id] ?? 0,
      labWithTotal: labWithTotal.length,
    };
  });

  const md: string[] = [];
  md.push("# OCR engine benchmark", "");
  md.push(`Run: ${new Date().toISOString().slice(0, 10)} · Node ${process.version} · ${labelled.length} labelled + ${unlabelled.length} unlabelled images${QUICK ? " (QUICK)" : ""}`, "");
  md.push("Images are downscaled to 1600 px (JPEG q80) first, exactly as the app will send them. `plain` = colour, `gray` = greyscale + contrast stretch. All configs use the same parser (`lib/ocr/parse-lines.ts`).", "");
  md.push("| config | total exact | total seen | items seen | items parsed | unlabelled: total line | unlabelled: consistent | p50 ms | p95 ms | init ms | errors |");
  md.push("|---|---|---|---|---|---|---|---|---|---|---|");
  for (const r of rows.sort((a, b) => b.totalExact - a.totalExact || b.itemsParsed - a.itemsParsed)) {
    md.push(
      `| ${r.id} | ${r.totalExact}/${r.truthWithTotal} (${pct(r.totalExact, r.truthWithTotal)}) | ${pct(r.totalSeen, r.truthWithTotal)} | ${pct(r.itemsSeen, r.annotated)} | ${pct(r.itemsParsed, r.annotated)} | ${pct(r.unlKeyword, r.unlN)} | ${pct(r.unlConsistent, r.unlN)} | ${r.p50.toFixed(0)} | ${r.p95.toFixed(0)} | ${r.init.toFixed(0)} | ${r.errors} |`,
    );
  }
  md.push("");
  md.push("**Columns.** *total exact*: the parser's grand total equals the annotated one (the metric that matters). *total seen*: the true total's digits appear somewhere in the raw OCR text (engine capability, ignoring the parser). *items seen / parsed*: share of annotated item prices found in the raw text / among the parser's items (annotated items are a lower bound, so treat as relative). *unlabelled*: no ground truth — *total line* = a TOTAL row with an amount was found; *consistent* = additionally the items sum to the receipt's own subtotal (a label-free correctness proxy).", "");
  md.push("**Caveats.** Node timings, not browser WASM. Labelled set is 20 mostly-US supermarket receipts; unlabelled PDFs are clean scans. Neither contains Malaysian mamak receipts, crumpled or dim phone photos — add your own to `bench/receipts/` with a `truth.csv` and re-run.", "");

  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, "ocr-bench.md"), md.join("\n") + "\n");
  fs.writeFileSync(path.join(OUT, "ocr-bench.json"), JSON.stringify({ rows, runs }, null, 1));
  console.log("\n" + md.slice(4, 7 + rows.length).join("\n"));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
