from safetensors.torch import load_file

from constants import MODEL_PATH
from unified_model_final import MultiTaskBert


class ConversationEvaluator:
    def __init__(self):
        self.model = self.load_model_from_checkpoint(MODEL_PATH)

    @staticmethod
    def load_model_from_checkpoint(model_path):
        model = MultiTaskBert()
        raw_state = load_file(model_path)
        infer_state = {k: v for k, v in raw_state.items() if not k.startswith("loss_fn_")}

        model.load_state_dict(infer_state)
        model.eval()

        return model

    def evaluate_turn(self, convo, turn, return_dict=False):
        model_output = self.model.predict(convo, turn)
        labels = self.model.decode_outputs(model_output, return_dict=return_dict)

        return labels

