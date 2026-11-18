#!/bin/bash
# docker/llm-server/entrypoint.sh
ollama serve &
sleep 5
# Modelfile : FROM /models/cardio-llm.gguf
if [ -f /models/Modelfile ]; then
  ollama create cardio-llm -f /models/Modelfile
fi
wait