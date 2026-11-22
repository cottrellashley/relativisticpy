// The editor of relativisticpy.live: the source on the left, and on the right each
// statement typeset with what came of it, computed as it is typed. Temml, which the
// Python side puts before this module, typesets; nothing is loaded from elsewhere.

/* global temml */

const PAUSE = 300; // milliseconds after the last key before the source runs

// What a selection, or a result, may be asked for: the action, and its label.
const ACTIONS = [
  ["compute", "Compute"],
  ["simplify", "Simplify"],
  ["expand", "Expand"],
  ["holds", "Does it hold?"],
  ["latex", "Copy LaTeX"],
];

const ASKED = {
  compute: "Computed",
  simplify: "Simplified",
  expand: "Expanded",
  holds: "Does it hold?",
  latex: "Its LaTeX",
};

function element(tag, className, text) {
  const made = document.createElement(tag);
  if (className) made.className = className;
  if (text !== undefined) made.textContent = text;
  return made;
}

// Typeset LaTeX into a node, as display mathematics; LaTeX Temml cannot read is shown as
// it is written.
function typeset(latex, node) {
  try {
    temml.render(latex, node, { displayMode: true, annotate: true, throwOnError: true });
    node.classList.add("rp-typeset");
  } catch (error) {
    node.textContent = latex;
    node.classList.add("rp-code");
  }
  return node;
}

function said(text) {
  return element("div", "rp-said", text);
}

// The error of a statement, or of a part: its headline, its notes and its hint.
function diagnostic(error) {
  const box = element("div", "rp-error");
  box.append(element("div", "rp-headline", `${error.kind}: ${error.message}`));
  for (const note of error.notes || []) box.append(said(`note: ${note}`));
  if (error.hint) box.append(said(`hint: ${error.hint}`));
  box.title = error.rendered || "";
  return box;
}

function definition(entry) {
  const defined = entry.definition;
  if (defined) return said(defined);
  return null;
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    return false;
  }
}

function render({ model, el }) {
  const view = Math.random().toString(36).slice(2, 8);
  let asked = 0; // how many requests this view has sent
  let running = null; // the id of the run this view waits for
  let shown = model.get("statements") || []; // the statements on show
  let stale = false; // whether the source has changed since they ran
  let committing = false; // whether this view is setting the model itself
  let committed = []; // what it set lately, which Jupyter echoes back, maybe after more typing
  let timer = null;

  // -- the layout ---------------------------------------------------------------------
  const root = element("div", "rp-live");
  // Keys typed in the editor are its own: the notebook neither runs its shortcuts on them
  // nor, as JupyterLab does inside a cell, keeps Enter from breaking a line.
  root.setAttribute("data-lm-suppress-shortcuts", "true");
  for (const type of ["keydown", "keypress", "keyup"]) {
    root.addEventListener(type, (event) => event.stopPropagation());
  }
  const bar = element("div", "rp-bar");
  const status = element("span", "rp-status", "");
  status.setAttribute("aria-live", "polite");
  const button = element("button", "rp-button", "Compute selection ▾");
  button.type = "button";
  button.title = "Compute the part of a statement selected in the source";
  bar.append(status, button);
  const panes = element("div", "rp-panes");
  const source = element("div", "rp-source");
  const backdrop = element("div", "rp-backdrop");
  backdrop.setAttribute("aria-hidden", "true");
  const input = element("textarea", "rp-input");
  input.spellcheck = false;
  input.setAttribute("autocapitalize", "off");
  input.setAttribute("autocomplete", "off");
  input.setAttribute("aria-label", "Mathematics, in LaTeX, plain text or Unicode");
  input.placeholder = "g_{\\mu\\nu} := Schwarzschild(M)\nR_{abcd} R^{abcd}";
  source.append(backdrop, input);
  const results = element("div", "rp-results");
  results.setAttribute("aria-label", "Results");
  panes.append(source, results);
  const menu = element("div", "rp-menu");
  menu.setAttribute("role", "menu");
  menu.hidden = true;
  const tip = element("div", "rp-tip");
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  root.append(bar, panes, menu, tip);
  el.append(root);

  // -- the source -----------------------------------------------------------------------
  function fit() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight + 2}px`;
  }

  // The text again, behind the input, with the span of each error marked.
  function mark() {
    const text = input.value;
    const marks = [];
    shown.forEach((entry, index) => {
      if (!entry.error || stale) return;
      let start = entry.error.start;
      let end = entry.error.end;
      if (!(end > start)) {
        start = entry.start;
        end = entry.end;
      }
      if (end > start && end <= text.length) marks.push([start, end, index]);
    });
    marks.sort((a, b) => a[0] - b[0]);
    backdrop.textContent = "";
    let done = 0;
    for (const [start, end, index] of marks) {
      if (start < done) continue;
      backdrop.append(document.createTextNode(text.slice(done, start)));
      const marked = element("mark", "rp-mark", text.slice(start, end));
      marked.dataset.entry = String(index);
      backdrop.append(marked);
      done = end;
    }
    backdrop.append(document.createTextNode(text.slice(done) + "​"));
  }

  function send(message) {
    asked += 1;
    const id = `${view}:${asked}`;
    model.send({ ...message, id });
    return id;
  }

  function run() {
    clearTimeout(timer);
    timer = null;
    running = send({ type: "run", source: input.value });
    root.classList.add("rp-running");
    status.textContent = "computing…";
  }

  function edited() {
    fit();
    stale = true;
    mark();
    results.classList.add("rp-stale");
    clearTimeout(timer);
    timer = setTimeout(run, PAUSE);
  }

  input.addEventListener("input", edited);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      run();
    } else if (event.key === "Escape") {
      close();
    }
  });

  // -- the results ------------------------------------------------------------------------
  function select(entry) {
    input.focus();
    input.setSelectionRange(entry.start, entry.end);
  }

  function show(statements, notes) {
    shown = statements;
    stale = timer !== null;
    results.textContent = "";
    results.classList.toggle("rp-stale", stale);
    statements.forEach((entry, index) => {
      const block = element("section", "rp-statement");
      block.dataset.index = String(index);
      if (entry.error) block.classList.add("rp-failed");
      const written = element("div", "rp-written");
      if (entry.read === false) {
        written.textContent = entry.statement;
        written.classList.add("rp-code");
      } else {
        typeset(entry.typeset || entry.statement, written);
      }
      written.title = `line ${entry.line}: select it in the source`;
      written.addEventListener("click", () => select(entry));
      block.append(written);
      (entry.values || []).forEach((value, which) => {
        const shownValue = typeset(value.latex, element("div", "rp-value"));
        shownValue.dataset.value = String(which);
        shownValue.addEventListener("contextmenu", (event) => {
          event.preventDefault();
          open(event.clientX, event.clientY, { entry, value });
        });
        block.append(shownValue);
      });
      const defines = definition(entry);
      if (defines) block.append(defines);
      for (const note of entry.notes || []) block.append(said(`note: ${note}`));
      if (entry.error) block.append(diagnostic(entry.error));
      block.append(element("div", "rp-answers"));
      results.append(block);
    });
    for (const note of notes || []) results.append(said(`note: ${note}`));
    const failed = statements.filter((entry) => entry.error).length;
    status.textContent = failed
      ? `${failed} of ${statements.length} statements failed`
      : `${statements.length} statement${statements.length === 1 ? "" : "s"}`;
    mark();
  }

  function answered(answer) {
    const holder = [...results.querySelectorAll(".rp-statement")].find((block) => {
      const entry = shown[Number(block.dataset.index)];
      return entry && entry.start <= answer.start && answer.end <= entry.end;
    });
    const target = holder ? holder.querySelector(".rp-answers") : results;
    const box = element("div", "rp-answer");
    const head = element("div", "rp-asked");
    head.append(element("span", "rp-action", ASKED[answer.action] || answer.action));
    const dismiss = element("button", "rp-dismiss", "×");
    dismiss.type = "button";
    dismiss.setAttribute("aria-label", "Dismiss");
    dismiss.addEventListener("click", () => box.remove());
    head.append(dismiss);
    box.append(head);
    if (answer.part) box.append(typeset(answer.part, element("div", "rp-part")));
    const judged = answer.action === "holds" && !answer.error && answer.values.length;
    if (answer.error) {
      box.append(diagnostic(answer.error));
    } else if (judged && answer.holds !== null) {
      const verdict = answer.holds ? "✓ holds" : "✗ does not hold";
      box.append(element("div", `rp-verdict ${answer.holds ? "rp-holds" : "rp-differs"}`, verdict));
    } else if (judged && !answer.notes.length) {
      box.append(element("div", "rp-verdict", "not decided"));
    }
    const verdictOnly = judged && answer.holds === true;
    for (const value of verdictOnly ? [] : answer.values || []) {
      box.append(typeset(value.latex, element("div", "rp-value")));
    }
    for (const note of answer.notes || []) box.append(said(note));
    if (answer.action === "latex" && answer.values && answer.values.length) {
      const latex = answer.values.map((value) => value.latex).join("\n");
      const code = element("pre", "rp-latex", latex);
      box.append(code);
      copy(latex).then((copied) => {
        box.append(said(copied ? "copied" : "select the LaTeX above to copy it"));
      });
    }
    target.append(box);
    box.scrollIntoView({ block: "nearest" });
  }

  // -- the menu -------------------------------------------------------------------------
  function close() {
    menu.hidden = true;
    tip.hidden = true;
  }

  function place(node, x, y) {
    const frame = root.getBoundingClientRect();
    node.style.left = `${Math.max(0, x - frame.left)}px`;
    node.style.top = `${Math.max(0, y - frame.top)}px`;
  }

  function ask(action, start, end) {
    if (timer !== null) run();
    send({ type: "compute", source: input.value, start, end, action });
  }

  // A menu for a selection of the source, or for a result: a result's actions are those
  // of its statement, and copying gives the result's own forms.
  function open(x, y, about) {
    menu.textContent = "";
    for (const [action, label] of ACTIONS) {
      const item = element("button", "rp-item", label);
      item.type = "button";
      item.setAttribute("role", "menuitem");
      item.dataset.action = action;
      item.addEventListener("click", () => {
        close();
        if (about.value && action === "latex") {
          copy(about.value.latex);
        } else if (about.value) {
          ask(action === "latex" ? "compute" : action, about.entry.start, about.entry.end);
        } else {
          ask(action, about.start, about.end);
        }
      });
      menu.append(item);
    }
    if (about.value) {
      const item = element("button", "rp-item", "Copy text");
      item.type = "button";
      item.setAttribute("role", "menuitem");
      item.dataset.action = "text";
      item.addEventListener("click", () => {
        close();
        copy(about.value.text);
      });
      menu.append(item);
    }
    place(menu, x, y);
    menu.hidden = false;
    const first = menu.querySelector("button");
    if (first) first.focus();
  }

  input.addEventListener("contextmenu", (event) => {
    if (input.selectionStart === input.selectionEnd) return;
    event.preventDefault();
    open(event.clientX, event.clientY, { start: input.selectionStart, end: input.selectionEnd });
  });
  button.addEventListener("click", () => {
    const frame = button.getBoundingClientRect();
    if (input.selectionStart === input.selectionEnd) {
      status.textContent = "select part of a statement in the source first";
      return;
    }
    open(frame.left, frame.bottom, { start: input.selectionStart, end: input.selectionEnd });
  });
  menu.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      close();
      input.focus();
    }
  });
  const away = (event) => {
    if (!menu.hidden && !event.composedPath().includes(menu)) close();
  };
  document.addEventListener("pointerdown", away, true);

  // -- errors under the pointer ------------------------------------------------------------
  input.addEventListener("pointermove", (event) => {
    const where = el.getRootNode();
    const at = (where.elementsFromPoint ? where : document).elementsFromPoint(
      event.clientX,
      event.clientY,
    );
    const marked = at.find((node) => node.classList && node.classList.contains("rp-mark"));
    const entry = marked ? shown[Number(marked.dataset.entry)] : null;
    if (!entry || !entry.error) {
      tip.hidden = true;
      return;
    }
    tip.textContent = "";
    tip.append(diagnostic(entry.error));
    place(tip, event.clientX + 12, event.clientY + 16);
    tip.hidden = false;
  });
  input.addEventListener("pointerleave", () => {
    tip.hidden = true;
  });

  // -- what Python says ----------------------------------------------------------------------
  function heard(message) {
    if (message.type === "ran" && message.id === running) {
      running = null;
      root.classList.remove("rp-running");
      show(message.statements, message.notes);
      if (timer !== null) return; // more has been typed: its run will be committed
      committed = [...committed.slice(-8), message.source, JSON.stringify(message.statements)];
      committing = true;
      try {
        model.set("source", message.source);
        model.set("statements", message.statements);
        model.save_changes();
      } finally {
        committing = false;
      }
    } else if (message.type === "computed" && message.id.startsWith(`${view}:`)) {
      answered(message);
    }
  }

  // The source, or the statements, set from Python, and not this view's own come back.
  function changed() {
    const text = model.get("source");
    if (committing || text === input.value || committed.includes(text)) return;
    input.value = text;
    fit();
    run();
  }

  function arrived() {
    const statements = model.get("statements") || [];
    if (committing || running !== null || committed.includes(JSON.stringify(statements))) return;
    show(statements, []);
  }

  model.on("msg:custom", heard);
  model.on("change:source", changed);
  model.on("change:statements", arrived);

  input.value = model.get("source") || "";
  fit();
  show(shown, []);
  run();

  return () => {
    clearTimeout(timer);
    document.removeEventListener("pointerdown", away, true);
    model.off("msg:custom", heard);
    model.off("change:source", changed);
    model.off("change:statements", arrived);
  };
}

export default { render };
