# results/tables

Written by `htrbench aggregate` from `results/scores.parquet` and the stored judge verdicts. All rates are
micro-averages over pages, at normalisation tier L1 unless stated. Definitions: `docs/METHODS.md`.

## profile.csv: one row per system

| Column | Meaning |
|---|---|
| `system_id`, `label`, `family`, `weights`, `role` | from `configs/systems.yaml`; `role` is `lineup` for the eight systems the poster follows through all dimensions |
| `n_pages`, `n_failed` | pages the system was run on; of these, pages without usable output (refusal, empty, error) |
| `char_accuracy`, `word_accuracy`, `fidelity`, `integrity`, `efficiency` | D1 to D5, each between 0 and 1 |
| `cer_page`, `wer_page` with `_lo`, `_hi` | whole-page error rates behind D1 and D2, with the 95 % bootstrap interval over pages |
| `glyph_f1`, `punct_kept`, `caps_kept` | the facets of D3; `glyph_f1` is measured on the Ratsprotokolle only |
| `numeral_f1`, `judge_misleading` | the facets of D4: numeral F1, and the share of judged lines rated misleading over both passes (D4 uses 1 − this). Only the eight line-up systems were judged; for the other twelve `judge_misleading` is empty and `integrity` is the numeral facet alone |
| `line_recall` | share of reference lines that received output text |
| `real_word_share` | share of word substitutions whose output token is a real word |
| `efficiency_speed`, `efficiency_hardware`, `efficiency_price` | the facets of D5 |
| `s_per_page`, `hardware_class` | median seconds per page; the smallest machine that runs one page |
| `price_usd_per_page`, `price_basis` | what the price facet was computed from: `list price`, `credits` or `electricity` |
| `wh_per_page`, `wh_basis` | energy per page of systems that are not hosted, and whether it was measured or estimated |

## profile_corpus.csv: one row per system and corpus

The columns of `profile.csv` computed on one corpus, with two differences. `integrity` is the numeral facet alone,
because the judged sample is too small to split, so there is no `judge_misleading` column. `efficiency` uses the
corpus' own median seconds per page with the system's hardware and price.

## judge.csv: one row per judged system

| Column | Meaning |
|---|---|
| `n_lines`, `n_ref_words` | judged lines and the words of their reference lines |
| `<type>_per_100w` | differences of each of the judge's ten types per 100 reference words, first pass |
| `share_usable`, `share_needs_correction`, `share_misleading` | verdicts of the first pass; the poster's panel *A reader's view* shows these |
| `share_misleading_both_passes` | the share that enters D4 |
| `kappa_verdict` | Cohen's κ between the two passes on the verdict |

## corpora.csv, runs.csv

`corpora.csv` gives per corpus the pages, the reference lines, the reference words and the pages the judge's
sample was drawn from. `n_lines` is the line count curated with the corpus, the number the poster prints;
`n_lines_scored` is the non-empty lines of the reference text, the unit the line matcher works on. The two differ
on 46 of the 86 pages, where the transcription holds a blank or a joined line.

`runs.csv` gives the archived runs with their model id, hardware, dates and notes.
