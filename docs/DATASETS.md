# Data

Release 1.0 is the material of the poster: 86 pages from four corpora of German handwriting, from the 18th to the
20th century, with 2,633 reference lines; the archived output of 20 systems on them; and the verdicts of the LLM
judge. It is one zip on Google Drive (`configs/data.yaml`), fetched with `htrbench data fetch` and checked file by
file against `SHA256SUMS`.

```
data/
  manifest.jsonl                 one page per line: reference text, facets, image checksum (schema.Page)
  pages/<corpus>/<page_id>.*     page images
  layout/<page_id>.xml           the Kraken segmentation every line engine read (PAGE-XML)
  runs/<system_id>__r0/          meta.json (schema.RunMeta) and predictions.jsonl (schema.Prediction)
  judge/gemini-3.5-flash/        <system_id>.jsonl: one row per judged line and pass
```

## Corpora

| Corpus | Pages | Lines | Material | Reference text | Holder and rights |
|---|---|---|---|---|---|
| `abp` Ratsprotokolle | 34 | 870 | council minutes of the prince-bishopric of Passau, series B5, 1736–1803; Kurrent with Latin formulae in Antiqua; numbered entries with marginal decisions | diplomatic transcription by project staff: ſ, abbreviation hooks, superscript vowels and line-final hyphens as written | Archiv des Bistums Passau. Images used by permission for research and not published online |
| `humboldt` travel journals | 20 | 602 | Alexander von Humboldt, England 1790 (14 pages) and America 1799 (6 pages); Kurrent with passages in Latin script, several languages, marginal notes and calculations | the TEI of *edition humboldt digital* (BBAW), flattened per page: deletions and editorial notes dropped, abbreviations left unexpanded, marginal notes after the body | BBAW, CC BY-SA 4.0. Text and facsimiles have been online since 2019, so language models may have seen them |
| `forst` forest operates | 22 | 823 | Bavarian *Forsteinrichtungsoperate*, 1857–1895: reports in a regular late Kurrent, 7 pages with tables | machine transcription corrected by hand; a table row is one line, its cells separated by spaces | Staatsarchiv Landshut, Regierung von Niederbayern, Kammer der Forsten. Unpublished |
| `laubmann` journals | 10 | 338 | ornithological field journals of Alfred Laubmann, 1917–1965, volumes 1 to 4 and one page of the register volume; a 20th-century hand with typed inserts | machine transcription corrected by hand | Bayerisches Hauptstaatsarchiv, GDA, Nachlass Alfred Laubmann. Unpublished |

The corpus names are those of the code and tables. `manifest.jsonl` carries per page the curated facets (genre,
period, script, languages, layout, condition, kind of scan) and a note where a page has a history. It also carries
a `year` for each Ratsprotokolle and Humboldt page, and none for the forest operates and the Laubmann journals,
which are dated per volume rather than per page.

**The date ranges printed on the poster are the archival series', not the release's.** The poster's corpus cards
give the span of the series the pages were drawn from — Ratsprotokolle 1710–1803, Humboldt 1790–1805 — while the
86 pages here run from 1736 to 1803 and 1790 to 1799. `htrbench check-poster` requires every page the manifest
dates to fall inside the printed range.

## How the references were made, and what follows from it

* **The forst and Laubmann references are post-edited Gemini output.** A Gemini model transcribed the pages in
  earlier projects and a person corrected the result against the image. Wherever the corrector let a defensible
  reading stand, such a reference is closer to a Gemini reading than an independent transcription would be, which
  favours the Gemini systems on these two corpora. Six of the ten Laubmann pages were corrected but not given a
  final check; the manifest notes which.
* **Only the Ratsprotokolle reference records letter forms.** The glyph facet of D3 exists for this corpus alone
  (`docs/METHODS.md`).
* **Humboldt is online, the other three are not.** Contamination is possible for one corpus and unlikely for three.
* **Reference lines.** Ratsprotokolle: the lines of the page transcription. Humboldt: the edition's `<lb/>` breaks
  and the ends of blocks. Forst and Laubmann: the lines of the corrected transcription. Line breaks do not enter D1
  and D2; they define the unit of the line-level measures.

## How the 86 pages came about

Pages were drawn per corpus with a fixed seed from the pages that had a reference: for the Ratsprotokolle evenly
over three half-centuries, for Humboldt from two journals. The working set held 107 pages.

1. Seven leaves of Humboldt's America journal (fol. 7r, 19r, 24r, 27r, 29v, 31v, 35r) were set aside. On each of
   them the three best general-purpose systems averaged a line CER of 40 % or more; such a page measures the scan,
   not the systems.
2. Of the remaining 100 pages, 86 were read by all eight systems of the poster's line-up. Transkribus German Genius
   was run on a subset of pages to save credits, and the comparison is restricted to the pages every line-up
   system has read.

Release 1.0 consists of those 86 pages. The other pages and the runs on them are not part of it.

## Archived runs

`runs/<system_id>__r0/predictions.jsonl` holds per page the text as scored, the raw output, seconds, token counts,
cost and measured energy where they exist, and the error if the page failed. `meta.json` holds model id, provider,
prompt hash, parameters, hardware and the dates of the run. All runs were made between 3 and 6 September 2026. 16
systems read all 86 pages. Four read a subset for cost reasons (`n_pages` in the tables): Gemini 3.1 Pro, Gemini 3.5
Flash with thinking and Claude Opus 5 read 29 pages each, CHURRO 18. They are compared on whole-page accuracy only.

Three things about the runs that the numbers do not show:

* **Split spreads.** Five forst scans showed two facing pages. They were cut at the fold after most systems had
  read them as one image, and the stored output of those runs was divided between the two halves by matching its
  lines to the halves' reference lines. Eight of the 22 forst pages are such halves (`_L`, `_R`; the manifest gives
  the cut). Both Transkribus runs were made after the cut and read the halves as separate pages. The `notes` of a
  run's `meta.json` name the spreads that were divided.
* **Two machines.** olmOCR-2 and Qwen3-VL 8B read the Ratsprotokolle and Humboldt pages on a Colab A100 without a
  power log, and the forst and Laubmann pages on a Modal A100 with one. Model, weights, prompt and decoding were the
  same. Their energy per page is the mean over the pages that have a measurement.
* **Frozen records.** When the release was frozen, the run records were restricted to the 86 pages and keyed by
  page id; ids that named the platform a system first ran on were replaced by ids that name the model; the `notes`
  were rewritten to drop identifiers of the working repository; and the CHURRO run's note was corrected, because it
  claimed the benchmark prompt had been passed to the model when the adapter replaces it with the model card's own
  (`scripts/modal_runner.py`). Texts, timings, token counts, costs and energy are as recorded at run time.
* **`gpu_s`** repeats the wall clock of the models that ran on a GPU. Nothing scores it; it is kept as provenance.

## Judge verdicts

`judge/gemini-3.5-flash/<system_id>.jsonl` holds, per judged line and pass, the reference line, the machine's line,
the context the judge saw, its spans with type and severity, its verdict and note, the raw answer, token counts and
cost. `docs/METHODS.md` describes the sample. Only verdicts for lines on the 86 pages are included.

## Access

The zip holds page images that their holders have not published, and transcriptions made for research. The Drive
folder is therefore shared on request rather than openly: `configs/data.yaml` names the folder and the repository
whose issue tracker requests go to. With the zip downloaded by hand, `htrbench data fetch --from <zip>` does the
rest, checksums included. The Humboldt part can be rebuilt from the edition's public TEI and IIIF images under its
own licence.
