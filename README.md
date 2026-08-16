# LLM Playground — LangChain

A set of focused, standalone scripts exploring core LangChain building blocks: prompt templates, chains, memory, and structured output parsing. Written as a personal learning log rather than a single application — each file isolates one concept.

## What's here

| File | Concept |
|---|---|
| `lc_basic.py` | `PromptTemplate` + `LLMChain` — building reusable prompts with placeholders (`{history}`, `{input}`, `{system_prompt}`) |
| `lc_memory.py` | Conversation memory strategies — buffer, windowed, token-buffer, and summary-buffer memory, and when each is appropriate |
| `lc_parser.py` | Structured output parsing — using schemas to force LLM output into a defined format and parse it into a Python dict |

## Key takeaways (from `learnings.md`)

- `LLMChain.predict()` inserts variables into a template, pulls from memory, and sends the request in one call.
- Memory strategy is a real design decision: `ConversationBufferMemory` (full history) and `ConversationSummaryBufferMemory` (summarized, token-aware) trade off cost and context differently.
- Output parsers work by generating formatting instructions inside the prompt itself, then parsing the model's response against that schema.

## Tech stack

Python · LangChain

## Running locally

```bash
pip install -r requirements.txt
python lc_basic.py
```
