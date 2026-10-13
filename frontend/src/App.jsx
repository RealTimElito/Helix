import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ask,
  createCollection,
  createMonitor,
  deleteDocument,
  getCollections,
  getDocuments,
  getHealth,
  getMonitors,
  runMonitor,
  uploadDocument,
} from "./api";

const MODES = [
  { id: "brief", label: "Brief" },
  { id: "extract", label: "Extract" },
  { id: "risk", label: "Risks" },
  { id: "compare", label: "Compare" },
  { id: "qa", label: "Question" },
];

const THEME_KEY = "helix-theme";

function storedTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* Storage can be blocked; fall through to the system preference. */
  }
  return null;
}

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

const PROMPTS = [
  "What are the key technical claims, limitations, and open questions?",
  "Give a risk summary with prioritized actions.",
  "How do these approaches differ on evaluation methodology and reported results?",
];

const CITE_RE = /\[([A-Za-z0-9_-]+)\s+p\.(\d+)\]/;

function splitCitation(value) {
  const text = String(value || "");
  const match = text.match(CITE_RE);
  if (!match) return { text, cite: "" };
  return { text: text.replace(match[0], "").trim(), cite: match[0] };
}

function citedLine(text, citation, missing) {
  if (missing) return `- ${text}`;
  return `- ${text}${citation ? ` ${citation}` : ""}`;
}

function coverageSentence(result) {
  const coverage = result?.coverage;
  if (!coverage) return "";
  const faith = Math.round((result.faithfulness || 0) * 100);
  const grounded = coverage.claims_grounded ?? 0;
  const dropped = coverage.claims_dropped ?? 0;
  return (
    `${coverage.notes_checked} of ${coverage.notes_in_collection} notes checked. ` +
    `${grounded} claims grounded, ${dropped} dropped. ` +
    `${faith}% faithfulness means each shown line appears on its cited page.`
  );
}

function briefMarkdown(answer, result) {
  if (!answer) return "";
  const lines = [`# ${answer.task || "Helix brief"}`, ""];
  const coverage = coverageSentence(result);
  if (coverage) lines.push(coverage, "");
  if (answer.summary) lines.push(answer.summary, "");
  if (answer.decision) lines.push(`Decision: ${answer.decision}`, "");
  if (answer.notes?.length) {
    for (const note of answer.notes) {
      lines.push("", `## ${note.title || note.cite_id || "Note"}`);
      for (const claim of note.claims || []) {
        lines.push(citedLine(claim.text, claim.citation, claim.missing));
      }
      for (const item of note.limitations || []) {
        lines.push(citedLine(`Limitation: ${item.text}`, item.citation, item.missing));
      }
      for (const item of note.open_questions || []) {
        if (typeof item === "string") {
          const parts = splitCitation(item);
          lines.push(citedLine(`Open question: ${parts.text}`, parts.cite));
        } else {
          lines.push(citedLine(`Open question: ${item.text}`, item.citation, item.missing));
        }
      }
    }
  } else {
    for (const claim of answer.claims || []) {
      lines.push(`- ${claim.text} ${claim.citation || ""}`);
    }
  }
  for (const aspect of answer.aspects || []) {
    lines.push("", `## ${aspect.name}`);
    for (const item of aspect.observations || []) {
      lines.push(citedLine(item.text, item.citation));
    }
  }
  if (answer.risks?.length) {
    lines.push("", "## Risks");
    for (const risk of answer.risks) {
      lines.push(`- (${risk.severity}) ${risk.statement} Action: ${risk.action} ${risk.citation || ""}`);
    }
  }
  return lines.join("\n");
}

function Cite({ cite, active, onPick }) {
  if (!cite) return null;
  return (
    <button
      type="button"
      className={active === cite ? "cite on" : "cite"}
      onClick={() => onPick(cite)}
    >
      {cite}
    </button>
  );
}

export default function App() {
  const queryClient = useQueryClient();
  const [view, setView] = useState("ask");
  const [collectionId, setCollectionId] = useState("");
  const [mode, setMode] = useState("brief");
  const [question, setQuestion] = useState(PROMPTS[0]);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [activeCite, setActiveCite] = useState("");
  const [result, setResult] = useState(null);
  const [collectionName, setCollectionName] = useState("");
  const [topic, setTopic] = useState("agent reliability and grounding");
  const [errorOpen, setErrorOpen] = useState(false);
  const [theme, setTheme] = useState(() => storedTheme() || systemTheme());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  function chooseTheme(next) {
    setTheme(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* The attribute still updates for this visit. */
    }
  }

  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 4000 });
  const collections = useQuery({ queryKey: ["collections"], queryFn: getCollections });

  useEffect(() => {
    if (!collectionId && collections.data?.length) {
      setCollectionId(collections.data[0].id);
    }
  }, [collectionId, collections.data]);

  const documents = useQuery({
    queryKey: ["documents", collectionId],
    queryFn: () => getDocuments(collectionId),
    enabled: Boolean(collectionId),
  });
  const monitors = useQuery({
    queryKey: ["monitors", collectionId],
    queryFn: () => getMonitors(collectionId),
    enabled: Boolean(collectionId) && view === "monitors",
  });

  const refreshDocs = () => {
    queryClient.invalidateQueries({ queryKey: ["documents", collectionId] });
    queryClient.invalidateQueries({ queryKey: ["collections"] });
  };

  const createCollectionMut = useMutation({
    mutationFn: () => createCollection(collectionName),
    onSuccess: (row) => {
      setCollectionName("");
      setCollectionId(row.id);
      queryClient.invalidateQueries({ queryKey: ["collections"] });
    },
  });
  const uploadMut = useMutation({
    mutationFn: (file) => uploadDocument(collectionId, file),
    onSuccess: refreshDocs,
  });
  const deleteMut = useMutation({
    mutationFn: deleteDocument,
    onSuccess: refreshDocs,
  });
  const askMut = useMutation({
    mutationFn: () => ask(collectionId, { question, mode, document_ids: selectedDocs }),
    onSuccess: (data) => {
      setResult(data);
      setActiveCite("");
    },
  });
  const monitorMut = useMutation({
    mutationFn: () => createMonitor(collectionId, topic),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["monitors", collectionId] });
    },
  });
  const runMut = useMutation({
    mutationFn: runMonitor,
    onSuccess: (data) => {
      setResult(data);
      setView("ask");
      queryClient.invalidateQueries({ queryKey: ["monitors", collectionId] });
    },
  });

  const error = askMut.error || uploadMut.error || runMut.error || createCollectionMut.error;
  useEffect(() => {
    if (error) setErrorOpen(true);
  }, [error]);

  const visibleChunks = useMemo(() => {
    const chunks = result?.chunks || [];
    if (!activeCite) return chunks;
    return chunks.filter((chunk) => chunk.cite === activeCite);
  }, [result, activeCite]);

  function toggleDoc(id) {
    setSelectedDocs((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  function exportBrief() {
    const blob = new Blob([briefMarkdown(result?.answer, result)], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "helix-brief.md";
    link.click();
    URL.revokeObjectURL(url);
  }

  const answer = result?.answer || {};
  const docs = documents.data || [];
  const scopeLabel = selectedDocs.length === 0
    ? "Whole collection"
    : `${selectedDocs.length} selected`;
  const faith = Math.round((result?.faithfulness || 0) * 100);
  const showError = Boolean(error) && errorOpen;
  const wholeCollection = (mode === "brief" || mode === "extract") && selectedDocs.length === 0;
  const coverageLine = coverageSentence(result);

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <span className="mark" aria-hidden="true">H</span>
          <div>
            <strong>Helix</strong>
            <p>Grounded briefs</p>
          </div>
        </div>
        <nav aria-label="Primary">
          <button type="button" className={view === "ask" ? "active" : ""} onClick={() => setView("ask")}>Ask</button>
          <button type="button" className={view === "library" ? "active" : ""} onClick={() => setView("library")}>Library</button>
          <button type="button" className={view === "monitors" ? "active" : ""} onClick={() => setView("monitors")}>Monitors</button>
        </nav>
        <label className="field">
          Collection
          <select
            value={collectionId}
            onChange={(event) => {
              setCollectionId(event.target.value);
              setSelectedDocs([]);
              setResult(null);
            }}
          >
            {(collections.data || []).map((collection) => (
              <option key={collection.id} value={collection.id}>
                {collection.name} ({collection.documents})
              </option>
            ))}
          </select>
        </label>
        <div className="rail-foot">
          <div className="theme-switch" role="group" aria-label="Theme">
            <button
              type="button"
              className={theme === "light" ? "active" : ""}
              aria-pressed={theme === "light"}
              onClick={() => chooseTheme("light")}
            >
              Light
            </button>
            <button
              type="button"
              className={theme === "dark" ? "active" : ""}
              aria-pressed={theme === "dark"}
              onClick={() => chooseTheme("dark")}
            >
              Dark
            </button>
          </div>
          <p className="status">
            Model {health.data?.llm === "ready" ? "ready" : "warming"} · embed {health.data?.embed || "…"}
            {health.data?.adapter ? " · adapter" : ""}
          </p>
        </div>
      </aside>
      <main>
        <div className="column">
          {showError && (
            <div className="note" role="alert">
              <p>{error.message}</p>
              <button type="button" onClick={() => setErrorOpen(false)}>Dismiss</button>
            </div>
          )}
          {view === "library" && (
            <section>
              <header className="page-head">
                <h1>Library</h1>
                <p>Upload PDFs, notes, or images. Text and tables become citable chunks.</p>
              </header>
              <div className="toolbar">
                <form
                  className="inline"
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (collectionName.trim()) createCollectionMut.mutate();
                  }}
                >
                  <input
                    aria-label="New collection name"
                    value={collectionName}
                    onChange={(event) => setCollectionName(event.target.value)}
                    placeholder="New collection"
                  />
                  <button type="submit">Create</button>
                </form>
                <label className="file">
                  <input
                    aria-label="Upload a document"
                    type="file"
                    accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) uploadMut.mutate(file);
                      event.target.value = "";
                    }}
                  />
                  <span>{uploadMut.isPending ? "Indexing…" : "Add a file"}</span>
                </label>
              </div>
              {uploadMut.isPending && <p className="hint">Indexing the file…</p>}
              {documents.data?.length === 0 ? (
                <div className="quiet">
                  <strong>Nothing filed yet</strong>
                  <p>This collection has no documents. Add a PDF, note, or image to start citing it.</p>
                </div>
              ) : documents.data ? (
                <ul className="doc-list">
                  {docs.map((doc) => (
                    <li key={doc.id}>
                      <div>
                        <strong>{doc.title}</strong>
                        <span>{doc.cite_id} · {doc.source_type}</span>
                      </div>
                      <button type="button" className="texty" onClick={() => deleteMut.mutate(doc.id)}>Remove</button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </section>
          )}
          {view === "monitors" && (
            <section>
              <header className="page-head">
                <h1>Monitors</h1>
                <p>Watch a topic and draft a synthesis from the current collection.</p>
              </header>
              <form
                className="inline"
                onSubmit={(event) => {
                  event.preventDefault();
                  monitorMut.mutate();
                }}
              >
                <input aria-label="Topic" value={topic} onChange={(event) => setTopic(event.target.value)} />
                <button type="submit">Add watch</button>
              </form>
              {runMut.isPending && (
                <div className="loading" role="status">
                  <div className="loading-track" aria-hidden="true"><i /></div>
                  <p className="loading-title">Drafting the synthesis</p>
                  <p>This checks every note in the collection and quotes sentences from those pages.</p>
                </div>
              )}
              {monitors.data?.length === 0 && !runMut.isPending ? (
                <div className="quiet">
                  <strong>No watches yet</strong>
                  <p>Add a topic to keep a standing question over this collection.</p>
                </div>
              ) : monitors.data?.length ? (
                <ul className="doc-list">
                  {(monitors.data || []).map((monitor) => (
                    <li key={monitor.id}>
                      <div>
                        <strong>{monitor.topic}</strong>
                        <span>{monitor.last_run ? `Last run ${monitor.last_run}` : "Not run yet"}</span>
                      </div>
                      <button type="button" disabled={runMut.isPending} onClick={() => runMut.mutate(monitor.id)}>
                        {runMut.isPending ? "Drafting…" : "Run synthesis"}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : null}
            </section>
          )}
          {view === "ask" && (
            <section className="ask">
              <header className="page-head">
                <h1>Ask the collection</h1>
                <p>A whole-collection brief checks every note and shows only sentences that appear on a cited page. Other modes draft with the language model.</p>
              </header>
              <div className="composer">
                <div className="segmented" role="group" aria-label="Answer mode">
                  {MODES.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={mode === item.id ? "active" : ""}
                      aria-pressed={mode === item.id}
                      onClick={() => setMode(item.id)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
                <textarea
                  aria-label="Question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  rows={4}
                />
                <div className="composer-bar">
                  <details className="scope">
                    <summary>{scopeLabel}</summary>
                    <div className="scope-panel">
                      <p className="hint">
                        {mode === "compare"
                          ? "Select two notes to compare them. Leave them unchecked to search the whole collection."
                          : mode === "brief" || mode === "extract"
                            ? "Leave the notes unchecked to fact-check every note in this collection. Tick notes to limit the question to those files."
                            : "Leave the notes unchecked to search the whole collection, or tick the ones to use."}
                      </p>
                      {docs.length === 0 && <p className="hint">No documents in this collection.</p>}
                      {docs.map((doc) => (
                        <label key={doc.id}>
                          <input
                            type="checkbox"
                            checked={selectedDocs.includes(doc.id)}
                            onChange={() => toggleDoc(doc.id)}
                          />
                          <span>{doc.title}</span>
                        </label>
                      ))}
                    </div>
                  </details>
                  <label className="starter">
                    <span className="sr">Starter question</span>
                    <select
                      aria-label="Starter question"
                      value=""
                      onChange={(event) => {
                        if (event.target.value) setQuestion(event.target.value);
                      }}
                    >
                      <option value="">Starters</option>
                      {PROMPTS.map((prompt) => (
                        <option key={prompt} value={prompt}>{prompt}</option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="primary"
                    disabled={askMut.isPending || !question.trim()}
                    onClick={() => askMut.mutate()}
                  >
                    {askMut.isPending ? "Working…" : "Generate"}
                  </button>
                </div>
              </div>
              {askMut.isPending && (
                <div className="loading" role="status">
                  <div className="loading-track" aria-hidden="true"><i /></div>
                  <p className="loading-title">Working through sources</p>
                  <p>
                    {wholeCollection
                      ? "Checking every note in the collection and quoting sentences from those pages."
                      : "Retrieving pages, drafting, and checking citations. The first answer on CPU takes several minutes while the model loads. Later ones are quicker, but still not instant."}
                  </p>
                </div>
              )}
              {!askMut.isPending && !result && (
                <div className="quiet">
                  <strong>No brief yet</strong>
                  <p>Write a question, pick a mode, and generate. The answer stays tied to the pages Helix actually retrieved.</p>
                </div>
              )}
              {!askMut.isPending && result && (
                <div className="result">
                  <article className="brief">
                    <header className="brief-head">
                      <div>
                        <p className="kicker">{answer.task || "Helix brief"}</p>
                        <div className="meter" title="Share of shown lines that appear on the cited page">
                          <span>Faithfulness</span>
                          <span className="meter-track" aria-hidden="true">
                            <i style={{ width: `${faith}%` }} />
                          </span>
                          <strong>{faith}%</strong>
                        </div>
                      </div>
                      <button type="button" onClick={exportBrief}>Export Markdown</button>
                    </header>
                    {coverageLine && <p className="coverage">{coverageLine}</p>}
                    {answer.summary && <p className="summary">{answer.summary}</p>}
                    {answer.decision && <p className="decision"><span>Decision</span> {answer.decision}</p>}
                    {answer.answer && (
                      <p className="summary">
                        {answer.answer} <Cite cite={answer.citation} active={activeCite} onPick={setActiveCite} />
                      </p>
                    )}
                    {(answer.notes || []).length > 0 ? (
                      answer.notes.map((note) => (
                        <section className="note-group" key={note.cite_id || note.title}>
                          <h2>{note.title}</h2>
                          <ul className="items">
                            {(note.claims || []).map((claim, index) => (
                              <li key={`c-${index}`} className={claim.missing ? "gap" : undefined}>
                                <p>{claim.text}</p>
                                {!claim.missing && (
                                  <Cite cite={claim.citation} active={activeCite} onPick={setActiveCite} />
                                )}
                              </li>
                            ))}
                            {(note.limitations || []).map((item, index) => (
                              <li key={`l-${index}`} className={item.missing ? "gap" : undefined}>
                                <p className="line-kind">Limitation</p>
                                <p>{item.text}</p>
                                {!item.missing && (
                                  <Cite cite={item.citation} active={activeCite} onPick={setActiveCite} />
                                )}
                              </li>
                            ))}
                            {(note.open_questions || []).map((item, index) => {
                              const missing = typeof item !== "string" && item.missing;
                              const parts = typeof item === "string"
                                ? splitCitation(item)
                                : { text: item.text, cite: item.citation || "" };
                              return (
                                <li key={`q-${index}`} className={missing ? "gap" : undefined}>
                                  <p className="line-kind">Open question</p>
                                  <p>{parts.text}</p>
                                  {!missing && (
                                    <Cite cite={parts.cite} active={activeCite} onPick={setActiveCite} />
                                  )}
                                </li>
                              );
                            })}
                          </ul>
                        </section>
                      ))
                    ) : (
                      <>
                        {(answer.claims || []).length > 0 && (
                          <section>
                            <h2>Claims</h2>
                            <ul className="items">
                              {(answer.claims || []).map((claim, index) => (
                                <li key={index}>
                                  <p>{claim.text}</p>
                                  <Cite cite={claim.citation} active={activeCite} onPick={setActiveCite} />
                                </li>
                              ))}
                            </ul>
                          </section>
                        )}
                        {(answer.limitations || []).length > 0 && (
                          <section>
                            <h2>Limitations</h2>
                            <ul className="items">
                              {(answer.limitations || []).map((item, index) => (
                                <li key={`l-${index}`}>
                                  <p>{item.text}</p>
                                  <Cite cite={item.citation} active={activeCite} onPick={setActiveCite} />
                                </li>
                              ))}
                            </ul>
                          </section>
                        )}
                      </>
                    )}
                    {(answer.risks || []).length > 0 && (
                      <section>
                        <h2>Risks</h2>
                        <ul className="items">
                          {(answer.risks || []).map((risk, index) => (
                            <li key={`r-${index}`}>
                              <p className="severity">{risk.severity}</p>
                              <p>{risk.statement}</p>
                              {risk.action && <p className="action">Action: {risk.action}</p>}
                              <Cite cite={risk.citation} active={activeCite} onPick={setActiveCite} />
                            </li>
                          ))}
                        </ul>
                      </section>
                    )}
                    {(answer.aspects || []).map((aspect) => (
                      <section key={aspect.name}>
                        <h2>{aspect.name}</h2>
                        <ul className="items">
                          {(aspect.observations || []).map((item, index) => (
                            <li key={index}>
                              <p>{item.text}</p>
                              <Cite cite={item.citation} active={activeCite} onPick={setActiveCite} />
                            </li>
                          ))}
                        </ul>
                      </section>
                    ))}
                    {!(answer.notes || []).length && (answer.open_questions || []).length > 0 && (
                      <section>
                        <h2>Open questions</h2>
                        <ul className="plain">
                          {answer.open_questions.map((item) => <li key={item}>{item}</li>)}
                        </ul>
                      </section>
                    )}
                    {result.dropped?.length > 0 && (
                      <p className="dropped">{result.dropped.length} unsupported items were removed by the citation check.</p>
                    )}
                    <details className="log">
                      <summary>
                        Trajectory
                        <span>{(result.trajectory || []).length} steps</span>
                      </summary>
                      <ol>
                        {(result.trajectory || []).map((step, index) => (
                          <li key={index}>
                            <strong>{step.tool}</strong>
                            <span>{step.detail}</span>
                          </li>
                        ))}
                      </ol>
                    </details>
                  </article>
                  <aside className="sources">
                    <div className="sources-head">
                      <h2>Sources</h2>
                      {activeCite && (
                        <button type="button" className="texty" onClick={() => setActiveCite("")}>Show all</button>
                      )}
                    </div>
                    {visibleChunks.map((chunk) => (
                      <article key={chunk.cite + chunk.text.slice(0, 24)}>
                        <h3>{chunk.cite}</h3>
                        <p>{chunk.text}</p>
                      </article>
                    ))}
                    {visibleChunks.length === 0 && (
                      <p className="hint">No source text for that citation in the retrieved set.</p>
                    )}
                  </aside>
                </div>
              )}
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
