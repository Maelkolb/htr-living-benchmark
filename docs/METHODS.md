# Methods

What a system is given, what is scored, and how each number on the poster is defined. The constants are in
`configs/protocol.yaml`; the module that implements a definition is named with it.

## Five dimensions

Each dimension is an absolute score between 0 and 1, computed from the same runs on the same pages. There is no
aggregate over the five: a profile is read as a shape, not as a rank.

| | Dimension | The archive's question | Definition |
|---|---|---|---|
| D1 | Character accuracy | How much of the page came back right? | 1 − CER over the whole page text |
| D2 | Word accuracy | How many of the page's words came back right? | 1 − WER over the whole page text |
| D3 | Diplomatic fidelity | Is the text kept as written: letter forms, punctuation, capitals? | mean of three character-alignment facets |
| D4 | Content integrity | Do the mistakes change the meaning? Would a reader be misled? | mean of the judge facet and the numeral facet |
| D5 | Efficiency | What does a page cost in time, hardware and money? | speed / 2 + (hardware + price) / 4 |

## Page image in, page text out

Every system reads the whole page. Page readers (hosted vision-language models, Transkribus, open-weight models)
get the page image. Line engines (TrOCR, Kraken) get the lines of one shared layout analysis, Kraken's default
baseline segmentation, and their line outputs are joined in the segmentation's reading order. The segmentation
files the archived runs used are part of the data release (`data/layout/`), so the layout is an input of the
benchmark and not a property of a line engine.

**Prompt.** One prompt for every system that takes a prompt (`prompts/page_v1.txt`; its SHA-256 is stored with each
run). It asks for a diplomatic transcription, one output line per physical line, main text before marginalia.
CHURRO runs with the prompt of its model card; PaddleOCR-VL and PP-OCRv5 are pipelines without a prompt.

**Decoding.** Gemini models run at temperature 0 with the thinking level given in `configs/systems.yaml`, safety
filters off, up to 8,192 output tokens. Claude models accept no sampling parameters; they run with adaptive thinking
at low effort and up to 8,192 output tokens. Open-weight page readers decode greedily in bf16 with up to 2,048 new
tokens. TrOCR runs in fp16 with beam size 4.

**Resolution.** Hosted models receive the image as it is in the data release. Open-weight page readers receive it
re-encoded as JPEG with the longer side at most 1,800 px.

**Failures.** A refusal, a safety block, an empty output or a failure after five attempts counts as a page with no
output: all of its reference text is deleted text. `n_failed` in the tables counts such pages.

**Transkribus.** The pages were uploaded as one document, the default layout analysis was run once, then one model
at a time. The text of a page is the text of its lines in the reading order of the PAGE-XML export
(`runners/transkribus.py`).

## Normalisation tiers (`eval/normalize.py`)

A tier says what two transcriptions may differ in and still count as the same reading. The tiers are cumulative.

| Tier | Adds | Used for |
|---|---|---|
| L0 | Unicode NFC; markdown table syntax dropped; line breaks and runs of whitespace become one space | glyph facet of D3 |
| L1 | words broken across lines joined; editorial marks dropped (`[?]`, `[...]`) or unwrapped (`[word]`); ligatures expanded; ſ → s, ß → ss, uͤ → ü | D1, D2, capitals, numerals, real-word share, the judge's sample |
| L2 | lower case; v → u, j → i | punctuation facet of D3 |

L1 removes what carries no reading difference between a diplomatic reference and a system that was not trained on
its conventions. It keeps spelling, capitalisation and punctuation.

## D1 and D2: character and word accuracy (`eval/score.py`, `eval/aggregate.py`)

Reference and output of a page are normalised at L1, which joins each of them into one string in reading order, and
aligned once (Levenshtein).

```
CER = Σ_pages min(edits(reference, output), |reference|) / Σ_pages |reference|        D1 = 1 − CER
WER = the same over whitespace tokens                                                  D2 = 1 − WER
```

This is the convention of end-to-end evaluation in the ICDAR READ competitions, OCR-D's dinglehopper and the
Transkribus compare tool: omitted text costs its length, invented text costs its length, a line break costs nothing,
and no rule has to decide what counts as a missing line. Reading order does count, because it is part of the page
string.

The cap makes a runaway output, a loop of thousands of characters, cost one page and not several.

The definition charges text that the reference does not contain. Where an edition leaves out a page header or a
catchword, a system that transcribes it pays for the difference.

## Line matching (`eval/linematch.py`)

D1 and D2 need no lines. Fidelity, numerals, the real-word share and the judge do, and they need the same unit for
a line engine, which emits one line per segmented baseline, and a page reader, which chooses its own line breaks.
That unit is the reference line. Output lines are assigned to reference lines one to one by character similarity
with a mild preference for similar positions on the page (Hungarian algorithm; pairs under 0.35 similarity are
rejected). A reference line left over is then searched inside neighbouring output lines, for systems that joined
two lines, and a matched line is extended by an unmatched neighbour when that improves it, for systems that broke a
line in two. What remains is a missing reference line, which enters the line-level counts with an empty output, or
an extra output line, which does not enter them.

`line_recall` in the tables is the share of reference lines that received text. It belongs next to every
line-level figure: the open-weight page readers return text for 11 % to 74 % of the reference lines, the hosted
models, the Transkribus models and the line engines for 81 % to 96 %.

## D3: diplomatic fidelity (`eval/fidelity.py`)

The mean of three facets, each a count on the character alignment of the matched line pairs. No lexicon and no
spelling rules are involved.

* **Glyphs.** F1 over the period letter forms (ſ, the abbreviation hook ʆ, ÿ/Ÿ, æ, œ, ꝛ, ꝑ/ꝓ, ꝗ/ꝙ, ꝰ, ꝝ, the
  ſt ligature, superscript e, nasal strokes; `PERIOD_GLYPHS` in `eval/fidelity.py` is the exact set) at L0: recall is the share of the reference's glyphs that come back as
  themselves at their aligned place, which is the HCPR of Levchenko (2025); precision is the share of the output's
  glyphs that the reference has. Measured on the Ratsprotokolle only. The other references write a plain s, so a
  correctly read ſ would count against the system there.
* **Punctuation.** The share of the reference's punctuation marks and symbols that are still at their aligned
  place, at L2.
* **Capitals.** Of the reference's capital letters that the system read as the same letter, the share that came
  back as capitals, at L1. A capital read as another letter is a misreading and belongs to D1.

A facet that does not exist for a corpus is left out of the mean; it never counts as zero.

## D4: content integrity (`eval/judge.py`, `eval/fidelity.py`)

The mean of two facets.

* **Judge.** 1 − the share of judged lines rated *misleading*, over both passes.
* **Numerals.** F1 over the tokens that contain a digit, compared as multisets per matched line at L1. Dates, sums
  and entry numbers are what registers are consulted for, and a wrong number is still a number.

**The judge.** Gemini 3.5 Flash sees the reference line, the machine's line and the two neighbouring reference
lines, never the image (`prompts/judge_v1.txt`). It returns every difference with one of ten types (garble,
spelling or normalisation, abbreviation expansion, meaning-preserving substitution, meaning-changing substitution,
name error, number error, omission, hallucinated insertion, punctuation or case), a severity from 0 to 2, and a
verdict for the line: *usable*, *needs correction* or *misleading*. A misleading line is fluent and wrong: a reader
would trust a statement, a name or a number that the source does not contain.

The sample consists of matched lines of at least three words whose words differ from the reference at L1, from the
29 pages marked `judge_sample` in the manifest, the corpora in turn, and the same reference lines across the
systems sampled together wherever a system has them. 100 lines were drawn per system. For six systems the draw
preceded the final page set, and 68 to 88 of their lines lie on pages of this release; only those are used. Claude
Sonnet 5 and PaddleOCR-VL were judged later, on 100 lines of the final set. `judge.csv` gives the lines and
reference words per system.

The poster's panel groups the ten types into the six bars it prints: *garble* (garble), *spelling /
normalisation* (spelling_or_normalisation, abbreviation_expansion, punctuation_case), *meaning preserved*
(meaning_preserving_substitution), *meaning changed* (meaning_changing_substitution), *names & numbers*
(name_error, number_error) and *omitted / invented* (omission, hallucinated_insertion).

Every line is judged twice at temperature 0. Cohen's κ between the two passes on the verdict is 0.87 to 0.98,
depending on the system. The poster's panel *A reader's view* shows the first pass: the differences per 100
reference words by type and the share of misleading lines. The score uses both passes, which is why the two
percentages can differ by a point or two.

The judge has a known bias: it tends to call any different real word a change of meaning. Its verdicts are stored
with the data release, because the judge costs money and is not deterministic; the tables are built from the stored
verdicts.

Only the eight line-up systems were judged. For the other twelve `judge_misleading` in `profile.csv` is empty and
integrity is the numeral facet alone; the poster shows D4 for the line-up only.

The judged sample is too small to be split by corpus. In `profile_corpus.csv`, and in the poster's matrix of
systems by corpus, integrity therefore rests on the numeral facet alone for every system.

## D5: efficiency (`eval/efficiency.py`)

Three facets with fixed anchors, so that a new system or a new price list never rescales the systems measured
before.

| Facet | Input | Score |
|---|---|---|
| speed | median seconds per page, as a user waits for it | 1 s or faster scores 1, 100 s or slower scores 0, linear in log₁₀ between |
| hardware | the smallest machine that runs one page (`hardware_class` in `configs/systems.yaml`) | CPU 1.0, GPU 8 GB 0.8, GPU 24 GB 0.5, GPU 80 GB 0.2, hosted 0.1 |
| price | USD per page | 0.1 ¢ or less scores 1, 1 $ or more scores 0, linear in log₁₀ between |

`efficiency = speed / 2 + (hardware + price) / 4`: half for the time a page takes, half for what it costs, the cost
divided between the machine one has to own and the money one has to pay. The weights are the equal-variance
weights (OECD/JRC 2008) of the three facets over the poster's line-up, 0.49, 0.24 and 0.26, rounded and frozen.

Seconds per page are the client's wall clock for the APIs, network included; the wall clock of a batch divided
among its pages in proportion to their lines for the line engines, without the shared segmentation; the time of the
generation call for models on rented GPUs, without model loading; and the wall time of the whole recognition job
divided by its pages for Transkribus, which reports nothing finer.

The price of a hosted system is its list price, from the token counts of each call, or its credits at the rate
they were bought for (`configs/prices.yaml`). An open-weight model is priced by the electricity of its measured
energy, whatever GPU the benchmark rented to run it: the rental is a convenience of the benchmark, not a cost of the
model. Energy is the GPU board power, sampled twice a second with `nvidia-smi` during the recognition, which leaves
out CPU, memory and cooling. Kraken runs on the CPU, where no power is logged; its energy is estimated as 35 W times
its mean seconds per page, while the speed facet uses the median. Two models (olmOCR-2, Qwen3-VL 8B) ran partly on a machine without a power log; their energy
is the mean over the pages that have one. `profile.csv` names the basis of every price and energy figure.

Per corpus the speed facet uses the corpus' own median seconds per page; hardware and price are those of the system.

## Reported with the scores

* **Real-word share** (`eval/plausibility.py`). Reference and output words are aligned per matched line; of the
  substitutions that are more than a change of punctuation or case, the share whose output token is a word. A word
  is a number, an entry of the wordfreq lists for the languages curated for the page (Zipf frequency at least 2.0
  for German, 2.5 for French, 3.0 for English), or any word of the benchmark's reference texts, so that period
  spellings count. This is the basis of the poster's third take-home: a fluent reader's errors are words.
* **Line recall**, as above.

## Statistics

Every rate is a micro-average: sums over pages divided by sums over pages, never a mean of per-page rates.
Confidence intervals for CER and WER are 95 % percentile intervals of a bootstrap over pages (1,000 resamples,
seed 0). Systems that did not read all 86 pages carry their page count (`n_pages`); the poster compares them on
nothing but whole-page accuracy.

## Reproducing the tables

`htrbench score` turns the archived runs into one row of counts per page and run (`results/scores.parquet`,
`schema.Score`). `htrbench aggregate` forms every rate, score and interval from sums of those counts and writes
`results/tables/`. `htrbench check-poster` makes 316 comparisons between what the poster prints, listed in
`poster/numbers.yaml`, and the tables. Scoring is deterministic; on the data release the three commands reproduce
the committed scores and tables exactly, which `tests/test_release.py` asserts.
