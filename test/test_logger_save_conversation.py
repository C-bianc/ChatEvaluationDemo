import csv
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

# Import necessary modules
from app.evaluator import MessageEvaluation
from app.utils.logger import save_conversation_with_evaluation
from unified_model_final import PredictionResult


class TestSaveConversation(unittest.TestCase):
    def setUp(self):
        self.log_path= Path(__file__).parent.parent/ "logs"
        self.conv_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Create sample PredictionResult objects
        prediction1 = PredictionResult(
            dimension="Communicative_Intent", label="I", logits={"D":0.1, "I":0.8, "O": 0.1})
        prediction2 = PredictionResult(dimension="Output_Elicitation", label="Yes", logits={"Yes":0.2, "No":0.8})
        prediction3 = PredictionResult(dimension="Helpfulness", label="Helpful", logits={"Helpful":0.1, "Neutral":0.8, "Not helpful": 0.1})

        # Create sample MesageEvaluation objects
        self.test_conversation = [
            MessageEvaluation(
                turn_number=1,
                role="bot",
                content="Hello, how can I help you?",
                evaluation=[prediction1, prediction2, prediction3],
                seed_scores={"seed_total": 0.8, "seed_intent": 0.8, "seed_output": 0.2, "seed_helpful": 0.1},
                reason_for_bad=None
            ),
            MessageEvaluation(
                turn_number=2,
                role="user",
                content="I need help with Python.", evaluation=[prediction1, prediction2, prediction3],
            seed_scores={"seed_total": 0.8, "seed_intent": 0.8, "seed_output": 0.2, "seed_helpful": 0.1},
                reason_for_bad=None
            ),
        ]

    def tearDown(self):
        # remove the file after each test
        for file in os.listdir(self.log_path):
            if file.startswith("conversation_"):
                os.remove(os.path.join(self.log_path, file))

    def test_save_conversation_creates_file(self):
        """Test that the function creates a CSV file with the correct name format."""
        # Get the timestamp that would be used for the conversation ID

        # Call function with a specific filename in the test directory

        test_filename = os.path.join(self.log_path, f"conversation_{self.conv_id}.csv")

        save_conversation_with_evaluation(self.test_conversation, self.conv_id)

        # Check that the file was created
        self.assertTrue(os.path.exists(test_filename))

    def test_save_conversation_writes_correct_data(self):
        """Test that the function writes the correct data to the CSV file."""
        # Call function with a specific filename in the test directory
        test_filename = os.path.join(self.log_path, f"conversation_{self.conv_id}.csv")
        save_conversation_with_evaluation(self.test_conversation, self.conv_id)

        # Read the CSV file
        with open(test_filename, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        # Check header row
        self.assertEqual(
            [
                "conv_id",
                "turn",
                "author",
                "message",
                "intent",
                "prob_intent",
                "elicitation",
                "prob_elicit",
                "helpfulness",
                "prob_helpfulness",
                "seed_total",
                "seed_intent",
                "seed_output",
                "seed_helpful",
            ],
            rows[0],
        )

        # Check data rows
        self.assertEqual(len(rows), 3)  # Header + 2 messages

        # Check first message row
        self.assertEqual(rows[1][0], self.conv_id)  # conv_id
        self.assertEqual(rows[1][1], "1")  # turn_number
        self.assertEqual(rows[1][2], "bot")  # turn_number
        self.assertEqual(rows[1][3], "Hello, how can I help you?")  # turn_text
        self.assertEqual(rows[1][4], "I")  # intent label
        self.assertEqual(rows[1][5], "0.8")  # intent label

    def test_save_conversation_with_default_filename(self):
        """Test that the function creates a file with default filename when none is provided."""
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
        expected_file_pattern = f"conversation_{conv_id}.csv"

        # mock the open function to avoid actual file creation
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            save_conversation_with_evaluation(self.test_conversation)

        # Check that open was called with the correct path
        # Extract the file path from the call arguments
        called_path = mock_file.call_args[0][0]
        self.assertTrue(called_path.endswith(expected_file_pattern))
        self.assertIn("logs", str(called_path))


if __name__ == "__main__":
    unittest.main()
