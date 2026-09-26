"""Dataset contract: download True/False domain files from
config.data.tf_base_url using tf_filename_template, verify each
file sha256 against data/raw/manifest.json (fail loudly on any
mismatch), normalize records to {{statement, label, domain,
entity}}. The entity is the native subject field when present,
otherwise a heuristic + spaCy en_core_web_sm NER fallback (the
NER path is audited on 100 extractions). TriviaQA subset loading
with normalized string-match against aliases lives here too."""
