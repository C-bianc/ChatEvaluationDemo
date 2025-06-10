### DEFINE RULES FOR WHEN TO TRIGGER SEED COMPUTATION


## INTENT
"""
if after 4 turns, the bot has not been evaluated with I or D, trigger intent
"""
import re
from logging import getLogger

logger = getLogger(__name__)

## OUTPUT ELICITATION
"""
if after 3 bot turns, the bot has not elicited output (label == NO), trigger output elicitation
"""

## HELPFULNESS
"""
if , after querying llm to check if user needs help progressing the conversation,
trigger helpfulness
"""

# SEED
## Has 3 SUB SCORES


class Seed:
    def __init__(self, weight_intent=1, weight_output_elicitation=1, weight_helpfulness=1, weight_eliciting=1):
        # weights should be between 0 and 1

        self.l_min = 3
        self.l_max = 8

        self.weight_intent = weight_intent
        self.weight_output_elicitation = weight_output_elicitation
        self.weight_helpfulness = weight_helpfulness
        self.weight_eliciting = weight_eliciting

        self.total_seed = None
        self.intent_subscore = None
        self.output_elicitation_subscore = None
        self.helpfulness_subscore = None

    @staticmethod
    def compute_subscore_intent(intent_labels):
        """
        intent is the ratio of D and I labels among total turns
        """
        if len(intent_labels) == 0:
            return 0

        n_goal_intents = intent_labels.count("D") + intent_labels.count("I")
        return round(n_goal_intents / len(intent_labels), 2)

    @staticmethod
    def _tokenize_turn(sentence):
        processed_sentence = sentence.replace("'", " ")
        tokens = re.findall(r"\w+", processed_sentence)

        return tokens

    def compute_subscore_output_elicitation(self, user_replies, elicitation_labels):
        if len(elicitation_labels) == 0:
            return 0

        """
        elicit score is the weighted ratio between eliciting turns and total turns added to learner contribution
        divided by two

        learner contribution is comptuted as the average response length of user turns
        with L_min being the minimum response length and L_max the maximum response length
        {
        0 if ARL < L_min
        min ( (ARL / L_max), 1) otherwise
        }
        :return:
        """
        user_replies_lengths = [len(self._tokenize_turn(reply)) for reply in user_replies]

        # compute average response length of user turns
        ARL = sum(user_replies_lengths) / len(user_replies_lengths)

        if ARL < self.l_min:
            LC = 0  # penalize if user resp length too short
        else:
            LC = min((ARL / self.l_max), 1)  # avoid score distortion if user resp length already long enough

        # compute eliciting ratio
        ER = elicitation_labels.count("Yes") / len(elicitation_labels)

        sub_score = (ER * self.weight_eliciting + LC) * 0.5 if LC > 0 else ER * self.weight_eliciting
        logger.info(
            f"subscore output elicitation: {sub_score} (l_min = {self.l_min}, l_max = {self.l_max}, ARL = {ARL}, ER = {ER}, LC = {LC})"
        )

        return round(sub_score, 2)

    @staticmethod
    def compute_subscore_helpfulness(helpfulness_labels):
        if len(helpfulness_labels) == 0:
            return 0

        n_helpful = helpfulness_labels.count("Helpful")
        n_neutral = helpfulness_labels.count("Neutral")
        if n_neutral > 0:
            n_neutral /= 2  # adapted because if only neutral, then seed is 0 (if trigger helpfulness when user in difficulty, then better)

        aggregated_labels = n_helpful + n_neutral
        return round(aggregated_labels / len(helpfulness_labels), 2)


    def compute_total_seed(self, intent_labels, output_labels, helpfulness_labels, user_replies) -> dict:
        self.intent_subscore = self.compute_subscore_intent(intent_labels)
        self.output_elicitation_subscore = self.compute_subscore_output_elicitation(user_replies, output_labels)
        self.helpfulness_subscore = self.compute_subscore_helpfulness(helpfulness_labels)

        seed_total = (
            (self.weight_intent * self.intent_subscore)
            + (self.weight_output_elicitation * self.output_elicitation_subscore)
            + (self.weight_helpfulness * self.helpfulness_subscore)
        ) / 3
        return {
            "seed": round(seed_total,2),
            "seed_intent": self.intent_subscore,
            "seed_output": self.output_elicitation_subscore,
            "seed_helpful": self.helpfulness_subscore,
        }
