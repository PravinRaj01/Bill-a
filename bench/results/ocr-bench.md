# OCR engine benchmark

Run: 2026-09-26 · Node v24.16.0 · 20 labelled + 40 unlabelled images

Images are downscaled to 1600 px (JPEG q80) first, exactly as the app will send them. `plain` = colour, `gray` = greyscale + contrast stretch. All configs use the same parser (`lib/ocr/parse-lines.ts`).

| config | total exact | total seen | items seen | items parsed | unlabelled: total line | unlabelled: consistent | p50 ms | p95 ms | init ms | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| tesseract/gray | 12/18 (67%) | 89% | 58% | 38% | 28% | 20% | 585 | 986 | 323 | 0 |
| tesseract/gray-up | 11/18 (61%) | 83% | 69% | 38% | 28% | 20% | 603 | 1225 | 154 | 0 |
| paddle-v5-en/gray | 11/18 (61%) | 78% | 41% | 33% | 30% | 18% | 1193 | 1529 | 77 | 0 |
| tesseract/plain | 10/18 (56%) | 83% | 60% | 39% | 28% | 20% | 612 | 1089 | 178 | 0 |
| paddle-v5-en/plain | 10/18 (56%) | 83% | 41% | 34% | 30% | 18% | 1302 | 1700 | 92 | 0 |
| paddle-tiny/gray | 8/18 (44%) | 67% | 33% | 28% | 28% | 18% | 528 | 866 | 42 | 0 |
| paddle-tiny/plain | 8/18 (44%) | 67% | 27% | 26% | 30% | 18% | 711 | 1080 | 149 | 0 |
| paddle-small/gray | 8/18 (44%) | 67% | 33% | 24% | 28% | 18% | 1339 | 1820 | 74 | 0 |
| paddle-small/plain | 7/18 (39%) | 61% | 30% | 23% | 28% | 18% | 1336 | 1722 | 127 | 0 |

**Columns.** *total exact*: the parser's grand total equals the annotated one (the metric that matters). *total seen*: the true total's digits appear somewhere in the raw OCR text (engine capability, ignoring the parser). *items seen / parsed*: share of annotated item prices found in the raw text / among the parser's items (annotated items are a lower bound, so treat as relative). *unlabelled*: no ground truth — *total line* = a TOTAL row with an amount was found; *consistent* = additionally the items sum to the receipt's own subtotal (a label-free correctness proxy).

**Caveats.** Node timings, not browser WASM. Labelled set is 20 mostly-US supermarket receipts; unlabelled PDFs are clean scans. Neither contains Malaysian mamak receipts, crumpled or dim phone photos — add your own to `bench/receipts/` with a `truth.csv` and re-run.

