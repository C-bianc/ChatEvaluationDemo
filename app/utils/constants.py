# constants.py
import os
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = os.environ.get("MODEL_PATH")


LLMS = ["llama3.1:latest", "wizardlm2:7b", "openchat:7b", "mistral:latest", "llama3-chatqa:8b", "gemma3:4b-it-qat"]

ACTION_INTENT = "No learning goal"
ACTION_OUTPUT = "Not eliciting output"
ACTION_HELPFUL = "Not helpful"


CONFIG_FILE = "app/assets/prompt_prefix.json"
