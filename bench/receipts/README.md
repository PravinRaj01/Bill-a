# OCR benchmark receipts

Drop your receipt photos in this folder (jpg / jpeg / png / heic-converted-to-jpg).
Everything here except this README and `truth.example.csv` is gitignored, so the
photos never reach the repo.

## What I need

- **15-20 photos**, ideally a realistic mix: a couple of clean/flat ones, some
  crumpled or curved, some dim or taken at an angle, at least one faded thermal
  print, and a few different merchants (mamak, supermarket, cafe, etc.).
- Original phone photos are best (don't pre-shrink or crop; the app will do its
  own downscale).

## Ground truth (optional but makes the numbers real)

Copy `truth.example.csv` to `truth.csv` and add one row per photo with the
printed **grand total** and the **number of item lines**. That is enough to
measure whether each OCR engine gets the total and the item count right. Without
it I can still compare engines' raw text and speed, but not accuracy.

## Privacy

Cover or crop card numbers and phone numbers if you'd rather. Names are fine to
leave, they never leave this folder.

## Generated from the Kaggle datasets

`node bench/prepare-datasets.mjs` (re)builds, from `bench/kaggle_datasets/`:

- `kaggle-ocr-NN.jpg` + `truth.kaggle-ocr.csv` / `.json` here: 20 labelled photos
  (mostly US supermarket receipts). The **grand totals are reliable**; the
  `item_lines` counts are a LOWER BOUND (annotators skipped some lines, e.g.
  Trader Joe's is labelled 18 of ~22), so score item recall by price, not by count.
  Two rows have no total (no box / "175, 000" thousands-separator), left blank.
- `../receipts-unlabeled/pdf-*.jpg` + `manifest.csv`: 40 rendered PDF scans
  (restaurant / cafe / retail, several countries), NO ground truth. Use for
  speed, crashes, and "did it find a total line", not accuracy. They are clean
  high-contrast scans, so easier than real phone photos.

Everything generated is gitignored.
