### DEFINE RULES FOR WHEN TO TRIGGER SEED COMPUTATION


## INTENT
"""
if after 4 turns, the bot has not been evaluated with I or D, trigger intent
"""
import re

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
    def __init__(self,weight_intent=1, weight_output_elicitation=1, weight_helpfulness=1, weight_eliciting=1):
        # weights should be between 0 and 1

        self.l_min = 3
        self.l_max = 8

        self.weight_intent = weight_intent
        self.weight_output_elicitation = weight_output_elicitation
        self.weight_helpfulness = weight_helpfulness
        self.weight_eliciting = weight_eliciting

        self.total_seed = None
        self.intent_sub_score = None
        self.output_elicitation_sub_score = None
        self.helpfulness_sub_score = None

    @staticmethod
    def compute_subscore_intent(intent_labels):
        """
        intent is the ratio of D and I labels among total turns
        """
        n_goal_intents = intent_labels.count("D") + intent_labels.count("I")
        return round(n_goal_intents / len(intent_labels), 2)

    def compute_subscore_output_elicitation(self, history, elicitation_labels):
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
        user_replies = [msg["content"] for msg in history if msg["role"] == "user"]
        user_replies_lengths = [len(re.sub("'", " ", reply).split()) for reply in user_replies]

        # compute average response length of user turns
        ARL = sum(user_replies_lengths) / len(user_replies_lengths)

        if ARL < self.l_min:
            LC = 0 # penalize if user resp length too short
        else:
            LC = min((ARL / self.l_max), 1) # avoid score distortion if user resp length already long enough

        # compute eliciting ratio
        ER = elicitation_labels.count("Yes") / len(elicitation_labels)

        sub_score = 0.5 * (ER * self.weight_eliciting + LC)

        return round(sub_score, 2)

    @staticmethod
    def compute_subscore_helpfulness(helpfulness_labels):
        n_helpful = helpfulness_labels.count("Helpful")
        return round(n_helpful / len(helpfulness_labels), 2)

    def compute_total_seed(self, intent_subcore, output_subscore, helpfulness_subscore):
        total_seed =   (self.weight_intent * intent_subcore) \
                     + (self.weight_output_elicitation * output_subscore) \
                     + (self.weight_helpfulness * helpfulness_subscore)
        return round(total_seed / 3, 2)
