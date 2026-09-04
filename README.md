# AI Agent Personality Bias Analysis

This project analyzes conversational behavior of different AI agent personas by extracting agent turns from saved run data and scoring them across linguistic dimensions such as politeness, filler words, empathy, directness, and CTA behavior.

## Project purpose

The repository compares persona styles such as:

- Gentle Guide
- Easy Expert
- Calm Authority
- Charming Challenger
- Direct Doer

It evaluates each conversation and aggregates the results per persona into a summary table.

## Repository structure

```text
.
├── check_bias.py             # Main analysis script
├── normal_dataloading.py     # Alternate script for loading conversation JSON files
├── data/
│   ├── calm-authority/
│   ├── charming-challenger/
│   ├── direct-doer/
│   ├── easy-expert/
│   └── gentle_guide/
│       └── run-...json
├── results/
│   └── 2026_09_04.tsv
├── .gitignore
└── README.md
```

## Data format

Each file in `data/<persona>/` contains a JSON export of a conversation run. The analysis script expects a structure similar to:

```json
{
  "daten": {
    "inputs": {
      "messages": [
        { "id": ["...", "AIMessage"], "kwargs": { "content": "..." } }
      ]
    },
    "outputs": {
      "messages": [
        { "id": ["...", "AIMessage"], "kwargs": { "content": "..." } }
      ]
    }
  }
}
```

The script reads all `AIMessage` entries and extracts spoken agent text from the message content.

## Analysis logic

`check_bias.py` does the following:

1. Loads all JSON files from the `data` folders.
2. Extracts agent turns from input and output messages.
3. Runs heuristic language analysis for metrics such as:
   - filler words
   - politeness
   - reasoning markers
   - slang
   - empathy
   - call-to-action signals
   - question frequency
   - exclamation frequency
   - average turn length
4. Aggregates the results by persona.
5. Saves the summary to a TSV file in `results/`.

## Main output

The script writes a grouped-by-persona table to:

- `results/2026_09_04.tsv`

This file contains average values per persona across all run files for that persona.

## Dependencies

The project requires Python packages including:

```bash
pip install pandas numpy spacy
```

You will also need the German spaCy model:

```bash
python -m spacy download de_dep_news_trf
```

## Run the analysis

From the project root:

```bash
python check_bias.py
```

This will:

- read all personae data,
- compute the metrics,
- print the persona summary in the terminal,
- save the TSV output to `results/`.

## Notes

- The current analysis uses a rule-based heuristic approach rather than a trained classifier.
- The metric dictionaries and word lists are defined in the script and can be adjusted for tuning.
- `normal_dataloading.py` is a smaller helper script for loading run data and can serve as a reference for alternative data extraction workflows.

## Example usage

```bash
python check_bias.py
```

Then inspect the returned summary and the generated TSV file in `results/`.
