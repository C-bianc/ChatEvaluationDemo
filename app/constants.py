# constants.py
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "/home/cbianc/Documents/memoire_travail/memoire_codes/linguineo_convs/models/final_architecture-bert-base-uncased_final_architecture-finetuned/checkpoint-2500/model.safetensors"

# Label mappings
DIMENSIONS = ["Communicative_Intent", "Output_Elicitation", "Helpfulness"]
LABEL2ID = {
    "Communicative_Intent": {"D": 0, "I": 1, "O": 2},
    "Output_Elicitation": {"No": 0, "Yes": 1},
    "Helpfulness": {"Helpful": 0, "Neutral": 1, "Not helpful": 2},
}


ID2LABEL = {dim: {v: k for k, v in LABEL2ID[dim].items()} for dim in LABEL2ID}


LLMS = ["llama3.1:latest", "wizardlm2:7b", "openchat:7b", "mistral:latest", "llama3-chatqa:8b", "gemma3:1b"]


CONFIG_FILE = "app/prompt_prefix.json"
