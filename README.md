# LLM Council

**A general-purpose multi-model LLM council.** Several independent language
models answer the same user request, critique competing answers, and consolidate
a response through a separate judge, with an optional independent model audit.
The implementation is oriented toward free-tier API endpoints.

Council is **not research-only**. It can handle ordinary questions, technical
explanations, coding advice, troubleshooting, brainstorming, decisions,
comparisons, creative writing, professional drafts, and long-form documents.
Research-design expansion remains one supported use case.

AI reviewers can be wrong. Council does not independently verify facts or
execute generated code, and no model consensus constitutes proof of correctness.

## Installation

Python 3.12 or newer is recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Configure the API keys of the providers you intend to use. Providers with
unavailable free quotas may be skipped, including the judge or auditor.

## Simplest way: run `./start`

Create **exactly two nonempty UTF-8 `.txt` files** in the local `input/`
folder. Their filenames are your choice. Council reads them in alphabetical
order, ignoring letter case, and combines their text with one blank line.

~~~text
input/
  01_request.txt    # Any question, instructions or task
  02_context.txt    # Context, examples, constraints or supporting material
~~~

Then, from the repository root, execute:

~~~bash
./start
~~~

The launcher uses `.venv/bin/python` if available and otherwise `python3`.
It does not alter your terminal's activated environment. It defaults to full
Council mode with automatic task routing and a 600-second deadline.

When the input folder is used, each new run **overwrites** these two files:

- `output/combined_input.txt` — the exact text the models receive.
- `output/answer.txt` — the answer, reviewer warnings, and run diagnostics.

Unique SQLite/JSONL event logs stay in `runs/` to support inspecting and
resuming previous runs. Local inputs, outputs and logs are gitignored.

To preview the combined input at **zero API cost**:

~~~bash
./start --dry-run
~~~

## Example requests

For a general explanation, the first file could contain:

~~~text
Explain how TLS 1.3 establishes session keys, with a simple example.
~~~

For coding help:

~~~text
Examine the Python exception in the context file and provide a corrected
implementation, explaining the root cause.
~~~

For writing:

~~~text
Write a concise professional email based on the notes supplied in context.
~~~

For comparative decision-making:

~~~text
Compare the two options in the context file, state trade-offs, and recommend
one given my constraints.
~~~

The same Council handles all these requests without adding a scientific
research design unless the user specifically asks for one.

## Answer mode and document mode

The default is `--task-type auto`.

- Ordinary tasks run the **answer pipeline**: independent proposals, review,
  judge consolidation and an independent audit when available.
- A request with **five or more consecutively numbered section headings**
  automatically runs **document mode**, which drafts multiple sections,
  assembles them, and attempts independent review.
- For a long document without numbered headings, explicitly request document
  mode. For an ordinary task containing a long numbered list, force answer mode.

~~~bash
./start --task-type document
./start --task-type answer
~~~

Your earlier 20-section research-design request still selects document mode
automatically; other topics can use document mode too.

## Modes and review depth

~~~bash
./start --mode full               # multiple models, review, judge, audit
./start --mode standard           # same general-purpose pipeline
./start --mode quick              # single model, faster and less reviewed
./start --review-depth thorough   # adds revisions and independent re-review
./start --deadline 300            # shorten total deadline
~~~

The older `--mode research` name remains accepted as a compatibility alias,
not a restriction. Fast review is the default. Provider timeouts, cooldowns,
rate limits and partial output are reported, not hidden.

You may also submit a direct prompt through the CLI:

~~~bash
python -m council.cli --mode full --task-type answer "Explain DNS caching."
~~~

Direct CLI prompts print the report in the terminal. The stable `output/`
files are specifically for the two-file workflow.

## Diagnostics and development

~~~bash
python -m council.cli --mode full --provider-report
python -m council.cli --mode full --provider-report --probe
python -m council.cli --replay RUN_ID
python -m council.cli --resume RUN_ID --run-id NEW_ID
python -m unittest discover -s tests -v
~~~

`--probe` makes real small API calls; `--dry-run` does not.

The models cannot browse the live web, run code, validate external citations,
or provide independent mathematical proofs through this Council pipeline.
The independent audit is a model opinion only. Free-tier API availability
and scientific correctness are not guaranteed.

Saved run data and some historical regression/evaluation fixtures still
use names such as `research_result` and `--mode research`. New outputs
also expose `council_result` and `request`. Legacy names are preserved
for compatibility with old tests and saved events, not as a task limitation.
