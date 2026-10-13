# Helix demo script

This is the walkthrough for a recorded demo. The repository does not include a video file.

1. Start the API and the web interface (see the README).
2. Open the sample library. It contains the fourteen training notes, `HX-001` through `HX-014`. Held-out notes `HX-015` through `HX-020` are not in this index.
3. Ask: "What are the key technical claims, limitations, and open questions?"
   The brief checks every note in the collection and quotes sentences from the pages. Show the grouped notes, the citation buttons, the coverage line, and the faithfulness meter. Faithfulness means each shown line appears on the cited page.
4. Open the trajectory and point at plan, retrieve, the quoted brief, and the citation check. There is no repair step on this path.
5. Switch two notes on, choose Compare, and ask how their evaluation methodology and reported results differ.
6. Choose Risks and ask for a risk summary with prioritized actions. Mention any lines the checker removed.
7. Add a monitor for "retrieval latency and grounding" and run a synthesis.
8. Upload one of the PDFs in `data/samples` and ask the same brief question against that file.
9. Export the brief as Markdown.

Optional stills: the brief with citations visible, the source pane for one citation, and the monitor list after a run.
