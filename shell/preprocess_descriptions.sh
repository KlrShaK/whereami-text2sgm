#!/bin/bash
# Preprocess frame descriptions via GPT and cache parsed scene graphs.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# Caption JSON root (frame-*.json files with descriptions)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Requires OPENAI_API_KEY to be set in the shell environment.
# Example:
#   export OPENAI_API_KEY="sk-..."

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(fcf66d7b-622d-291c-86b8-7db96aebcee3)
SCENE_IDS=()



# • --embedding_mode controls how each text string gets converted into the *_word2vec vector fields.

#   - token: uses the first token’s spaCy vector (nlp(text)[0].vector) via get_word2vec (playground/graph_models/models/preprocess_descriptions.py:62, playground/graph_models/data_processing/
#     graph_loader_utils.py:118).
#   - doc: uses the whole text/doc vector (doc.vector) via create_embedding_nlp (playground/graph_models/models/preprocess_descriptions.py:54, playground/graph_models/data_processing/
#     create_text_embeddings.py:115).

#   Where it’s defined:

#   - CLI arg and choices: playground/graph_models/models/preprocess_descriptions.py:167
#   - Your shell wrapper currently sets token: shell/preprocess_descriptions.sh:19

#   So with relations like "next to", token mostly reflects "next", while doc reflects the whole phrase.

# Additional options (uncomment as needed)
EXTRA_ARGS=(
  --embedding_mode doc
  --workers 8 #32
  --batch_size 200
  # --dry_run
  # --overwrite
  # --max_frames 5
)

cd "$PROJECT_DIR" || exit 1

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "Set it with: export OPENAI_API_KEY='sk-...'"
  exit 1
fi

CMD=(
  python preprocess_descriptions.py
  --data_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
