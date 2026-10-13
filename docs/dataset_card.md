# Helix supervised dataset card

## Summary

Helix instruction data is built from an original library of twenty technical notes. Fourteen notes are used for training. Six notes (`HX-015` through `HX-020`) are held out entirely, including every page of those notes.

Gold answers are derived from the note metadata. Claim, limitation, method, result, risk, and decision strings are copied into the page text, so a correct answer is a source sentence with a citation.

## Counts

- Training examples: 2064
- Held-out examples: 94
- Training tasks: {"qa": 672, "brief": 624, "risk": 336, "extract": 336, "compare": 96}
- Held-out tasks: {"qa": 48, "extract": 12, "risk": 12, "compare": 10, "brief": 12}

## Tasks

- `extract`: claims, limitations, and open questions
- `brief`: decision brief with risks and citations
- `risk`: severity, statement, and sourced action
- `qa`: quote the supporting sentence
- `compare`: methodology and reported results, cited per document

## Construction

Question wording is templated. Training questions are also prefixed with two instruction wrappers so the model sees the same gold under slightly different requests. Held-out questions use wrappers that do not appear in training.

The notes are original. They are not excerpts from papers, advisories, or other copyrighted sources.

## Intended use

Supervised fine-tuning of an open instruct model for grounded technical briefs. The held-out split is for structured accuracy and preference comparisons against the base model.

## Limitations

The library is small and stylistically related, even though the held-out notes use different facts. Results describe this Helix task mixture, not general question answering.
