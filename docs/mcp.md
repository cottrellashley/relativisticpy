# The MCP server

> **For:** anyone who gives an AI agent mathematics to do, and agents themselves. **You
> will learn:** how to serve RelativisticPy to any MCP client, what each tool takes and
> gives, and the bounds the server keeps.
> **Before this:** [Getting started](getting-started.md).

Language models write physics and mathematics fluently, and get long algebra wrong.
RelativisticPy reads the LaTeX they write, computes it exactly, and says where a claim
does not hold. `relativisticpy mcp` serves it over the
[Model Context Protocol](https://modelcontextprotocol.io), on standard input and output,
as tools an agent calls:

| Tool | What it does |
|---|---|
| [`compute`](#compute) | runs a program, and gives each statement's values, what it defines, and its error, if it failed; a `session` keeps definitions from one call to the next |
| [`verify`](#verify) | checks one claimed equality or step of a derivation, `lhs = rhs` or `a = b = c`, after its context: holds, differs, relates, not decided or error, with the difference of the sides in LaTeX |
| [`check`](#check) | checks every display equation of a LaTeX document or Markdown notebook |
| [`help`](#help) | the documentation, offline; `reference` is the whole language on one page |
| [`list_sessions`, `reset_session`](#sessions) | the named sessions, and forgetting one |

Every page of this documentation is also a [resource](#resources), and two
[prompts](#prompts) start a verification and the check of a paper.

## Install

The server needs the `mcp` extra, which adds the official MCP SDK:

```bash
pip install "relativisticpy[mcp] @ git+https://github.com/cottrellashley/relativisticpy@v2"
```

`relativisticpy mcp` then waits for a client on standard input; a client starts it
itself, as below. `python -m relativisticpy mcp` is the same command.

## Connect a client

A client needs one thing: the command that starts the server. Give a client the absolute
path of `relativisticpy` (`which relativisticpy`) if it does not find it, since clients
start servers with an environment of their own.

### Claude Code

```bash
claude mcp add relativisticpy -- relativisticpy mcp
```

To let it check the papers of a folder by their paths, allow the folder:

```bash
claude mcp add relativisticpy -- relativisticpy mcp --allow-files /path/to/papers
```

`/mcp`, in a session, shows the server and its tools. Without installing anything first,
`uvx` runs it from GitHub:

```bash
claude mcp add relativisticpy -- uvx --from "relativisticpy[mcp] @ git+https://github.com/cottrellashley/relativisticpy@v2" relativisticpy mcp
```

### Claude Desktop

Add the server to `claude_desktop_config.json`, which is in
`~/Library/Application Support/Claude` on macOS and in `%APPDATA%\Claude` on Windows,
and restart Claude Desktop:

```json
{
  "mcpServers": {
    "relativisticpy": {
      "command": "relativisticpy",
      "args": ["mcp", "--allow-files", "/Users/you/papers"]
    }
  }
}
```

Or, run from GitHub by `uvx`:

```json
{
  "mcpServers": {
    "relativisticpy": {
      "command": "uvx",
      "args": [
        "--from",
        "relativisticpy[mcp] @ git+https://github.com/cottrellashley/relativisticpy@v2",
        "relativisticpy",
        "mcp"
      ]
    }
  }
}
```

### Any MCP client

Every client that starts servers over standard input and output takes the same command
and arguments. Cursor reads `.cursor/mcp.json`, in the same form as Claude Desktop;
VS Code reads `.vscode/mcp.json`:

```json
{
  "servers": {
    "relativisticpy": {
      "type": "stdio",
      "command": "relativisticpy",
      "args": ["mcp", "--time-limit", "120"]
    }
  }
}
```

### From Python

The SDK's own client connects to the server in memory, with no process of its own, as the
tests of this page do. `call` here calls one tool of a new server, and gives its
structured result:

```python
>>> import anyio
>>> from mcp import Client
>>> from relativisticpy.mcp import create_server
>>> def call(tool, **arguments):
...     async def run():
...         async with Client(create_server()) as client:
...             result = await client.call_tool(tool, arguments)
...             return result.structured_content
...     return anyio.run(run)

```

`Client(StdioServerParameters(command="relativisticpy", args=["mcp"]))` starts it as a
command instead, as a client does.

## The tools

Each tool says, in its description, what it takes and gives, and the limits the server
was started with; the server's instructions, which a client may give the model, sum up
the language.

### compute

Runs a program: mathematics as written, in LaTeX, plain text or Unicode, with statements
on lines of their own or separated by `;`.

| Argument | What it is |
|---|---|
| `source` | the program |
| `session` | a session to run in, made on first use, whose definitions persist from one call to the next; without it, the program runs in a new, empty workbook |
| `time_limit` | seconds the call may run, up to the server's limit |

It gives one entry for each statement, in order, exactly as a line of
`relativisticpy --format json` gives it (see [the command line](cli.md#forms-of-output)):
where it is, the statement, its `values`, each as `relativisticpy.display.data`
describes it, with its `text`, `unicode` and `latex` forms, what it `defines`, `notes` on
how it was read, and its `error`. `ok` says whether every statement ran and none failed.

```python
>>> result = call("compute", source=r"g_{\mu\nu} := Schwarzschild(M); R_{abcd} R^{abcd}")
>>> [entry["statement"] for entry in result["statements"]]
['g_{\\mu\\nu} := Schwarzschild(M)', 'R_{abcd} R^{abcd}']
>>> [d["name"] for d in result["statements"][0]["defines"]]
['t', 'r', 'theta', 'phi', 'g']
>>> kretschmann = result["statements"][1]["values"][0]
>>> kretschmann["text"], kretschmann["unicode"], kretschmann["latex"]
('48*M^2/r^6', '48M²/r⁶', '\\frac{48 M^{2}}{r^{6}}')
>>> result["ok"]
True

```

A statement that fails does not stop the others. Its error comes whole, as
`RelativisticPyError.to_dict()` gives it: its type, its message, where it is, the text it
is about, its notes and its hint:

```python
>>> result = call("compute", source="x := 2\n\\frac{1}{2\nx + 1")
>>> result["statements"][1]["error"]
{'type': 'ParseError', 'kind': 'ParseError', 'message': "unclosed '{' in the argument of \\frac", 'file': '<input>', 'line': 2, 'column': 9, 'end_line': 2, 'end_column': 10, 'text': '{', 'notes': [], 'hint': "add '}' where the argument ends"}
>>> result["statements"][2]["values"][0]["text"], result["ok"]
('3', False)

```

### verify

Checks one claimed equality, or one step of a derivation, as
[a check](checking.md) judges the equations of a paper.

| Argument | What it is |
|---|---|
| `claim` | an equation, `lhs = rhs`, or a chain, `a = b = c`, each link of which is judged |
| `context` | what the claim relies on, run first: definitions, a metric, coordinates, or symbols declared as variables, as `Reals := [x]` |
| `time_limit` | seconds the call may run, up to the server's limit |

The `verdict`, on the claim and on each of its `links`:

| Verdict | Meaning |
|---|---|
| `holds` | proven equal |
| `differs` | every quantity is known, and the sides are not equal: a mistake. The `difference`, left side minus right, is given in each form, LaTeX among them |
| `relates` | the sides are not equal, but involve quantities the context does not define, so the claim may be an equation to solve rather than an identity; the difference is given. Declaring the symbols in the context, as `Reals := [x]`, has such a claim judged |
| `not decided` | not proven equal, though numbers do not show that the sides differ; or a relation such as `\approx`; or over the time limit |
| `error` | the claim could not be read or computed, or is not a claim; its `detail`, place, `notes` and `hint` say why |

Each link also has `check`, the verdict `relativisticpy check` would give it. Statements of
the context that fail are listed in `context`. A step with its sign wrong:

```python
>>> step = call(
...     "verify",
...     claim=r"\frac{d}{dr} \left(1 - \frac{2M}{r}\right)^{-1} = \frac{2M}{(r - 2M)^2}",
...     context="Reals := [r, M]",
... )
>>> step["verdict"], step["links"][0]["difference"]["latex"]
('differs', '- \\frac{4 M}{4 M^{2} - 4 M r + r^{2}}')
>>> claim = call(
...     "verify",
...     claim=r"\Gamma^{r}_{\theta\theta} = -r",
...     context=r"g_{\mu\nu} := Schwarzschild(M)",
... )
>>> claim["verdict"], claim["links"][0]["detail"]
('differs', 'the left side minus the right is 2M')
>>> chain = call("verify", claim="(x+1)^2 = x^2 + 2x + 1 = x^2 + 1", context="Reals := [x]")
>>> [(link["verdict"], link["statement"]) for link in chain["links"]]
[('holds', '(x+1)^2 = x^2 + 2x + 1'), ('differs', 'x^2 + 2x + 1 = x^2 + 1')]

```

### check

Checks the equations of a LaTeX document or a Markdown notebook, as
[`relativisticpy check`](checking.md) does, and gives the report as `Report.to_dict()`
gives it.

| Argument | What it is |
|---|---|
| `document` | the text of the document |
| `path` | or the path of a file, which the server may read only in the folders it was started with, by `--allow-files`; relative to its working directory or to one of those folders |
| `format` | `latex`, the default for text, or `markdown`; a path goes by its suffix |
| `name` | the name findings in text are located by, as `paper.tex` |
| `lines` | only these lines, as `120:340`, after what comes before them has run, for its definitions |
| `strict`, `fail_on` | a policy: fail on what `--strict` fails on, or on these verdicts; `failed` says whether the document fails it |
| `all` | list every statement's finding; by default `findings` lists what fails `--strict` and what ran over its time limit, and `counts` counts the rest |
| `statement_time_limit` | seconds each statement may run (default 30) |
| `time_limit` | seconds the whole check may run, up to the server's limit; a check that runs out of time reports what it checked, and says where it stopped |

```python
>>> paper = r"""
... \begin{equation} ds^2 = -dt^2 + a(t)^2 dx^2 \end{equation}
... \begin{equation} \Gamma^x_{tx} = \frac{\dot a}{a} \end{equation}
... \begin{equation} \Gamma^t_{xx} = a \end{equation}
... """
>>> report = call("check", document=paper, name="paper.tex", strict=True)
>>> report["counts"], report["failed"]
({'holds': 1, 'differs': 1, 'defines': 1}, True)
>>> [(f["line"], f["verdict"], f["difference"]["text"]) for f in report["findings"]]
[(4, 'differs', "a(t)*a'(t) - a(t)")]

```

### help

The documentation, as Markdown, offline. Without a `topic`, what the language looks like
and the topics; `reference` is [the language reference](reference.md), the whole
language on one page of about 8,000 tokens, which an agent can read before it writes.
A topic may be asked for by another name, as `check` for `checking`.

## Sessions

`compute` with a `session` runs in the workbook of that name, so that a metric defined in
one call is there in the next; each call's statements are located as `<input 1>`,
`<input 2>` and so on. `list_sessions` gives each session's statements, what it defines,
and the server's limits; `reset_session` forgets one, with everything it defined. A
server keeps eight sessions at most (`--max-sessions`).

## Resources

Every page of this documentation is a resource, `relativisticpy://docs/{topic}`, as
Markdown: `relativisticpy://docs/reference` is the language reference, and
`relativisticpy://docs/check` the page on checking, by another name for it. A client can
put one in front of the model before it writes.

## Prompts

| Prompt | Arguments | What it asks for |
|---|---|---|
| `verify_derivation` | `derivation`, `context` | each step verified with `verify`, its verdict reported, and a wrong step corrected |
| `check_paper` | `document`, a path or LaTeX | the document checked with `check`, and each finding that fails explained, with a fix |

## Errors

What fails in a statement is that statement's `error`, and the call goes on. A call that
cannot be done at all is a tool error, `isError` in MCP, whose text says what, and how to
go on, in a line `hint:`: a session that cannot be named so, a file the server may not
read, a topic there is no help on, or a computation that would not stop at its time
limit.

## Safety

The server is for an agent to drive on its own, so it keeps these bounds:

- **No Python.** Workbooks are given no Python scope, and the language never runs Python:
  text in quotes is never evaluated, and `\texttt{os}` is a name. No tool binds data from
  a client, and no tool writes a file.
- **Files only where allowed.** `check` reads a file only if the server was started with
  `--allow-files`, and only in those folders, after symbolic links and `..` are followed.
  The files a document reads in, by `\input`, `\include` or `\usepackage`, are held to the
  same folders. Without the option, `check` takes documents as text.
- **Every call in time.** Each call runs in a worker process of its own: one for each
  session, and up to four for calls without one. A call may run for 60 seconds, or the
  server's `--time-limit`, and may ask for less. When the time runs out, the statement
  running is stopped and reported as over the time limit, with what ran before it, and a
  session keeps its definitions. A computation that does not stop when asked, as one in
  a library's compiled code may not, has its worker stopped five seconds later: the call
  fails, saying so, and a session computing in it is lost. A client that gives up on a
  call interrupts its computation.
- **Memory.** Each worker may use 4096 MiB (`--memory-limit`), where the system enforces
  it, as Linux does; a computation that needs more fails with `there is not enough memory
  to compute this`, and the server goes on.
- **Results of a size.** What `compute`, `verify` or `check` gives is cut to 50,000
  characters of JSON (`--max-output`): long text first, marked `…`, then lists, with
  `truncated` set and a note that says what was cut. A program, claim or document of
  more than 1,000,000 characters is refused.
- **Workers of their own.** A worker starts in the folder RelativisticPy is installed in,
  with Python's safe path, so that no module in the server's working directory can stand
  in for one RelativisticPy imports. It ends when the server does.

On Windows, which has no `SIGALRM`, a statement is not stopped softly: its worker is
stopped instead, at the end of the call's time; nor is the memory limit enforced there.

## Options

| Option | What it does |
|---|---|
| `--allow-files [FOLDER]` | lets `check` read documents in FOLDER, and the files they include there; without FOLDER, the working directory. Give it again for more folders |
| `--time-limit SECONDS` | the longest any call may run (default 60); a call may ask for less |
| `--max-output CHARACTERS` | the most characters a result may have, as JSON (default 50000) |
| `--max-sessions N` | the most named sessions at once (default 8) |
| `--memory-limit MIB` | the memory each worker may use, where the system enforces it; 0 for no limit (default 4096) |

`relativisticpy mcp --help` lists them, with examples.

**Next:** [the language reference](reference.md), for what an agent may write, or
[checking papers](checking.md), for what each verdict means.
