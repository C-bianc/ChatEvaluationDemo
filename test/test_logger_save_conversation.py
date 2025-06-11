import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

# Import necessary modules
from app.evaluator import ConversationEvaluator
from app.utils.logger import save_conversation_with_evaluation
from unified_model_final import PredictionResult, MultiTaskBert


class TestSaveConversation(unittest.TestCase):
    def setUp(self):
        self.log_path = Path(__file__).parent.parent / "logs"
        self.conv_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Mock the model required by ConversationEvaluator
        mock_model = mock.Mock(spec=MultiTaskBert)

        # Create a ConversationEvaluator instance
        self.evaluator = ConversationEvaluator(evaluation_model=mock_model)

        # Create sample PredictionResult objects
        prediction1 = PredictionResult(
            dimension="Communicative_Intent", label="I", logits={"D": 0.1, "I": 0.8, "O": 0.1}
        )
        prediction2 = PredictionResult(dimension="Output_Elicitation", label="Yes", logits={"Yes": 0.2, "No": 0.8})
        prediction3 = PredictionResult(
            dimension="Helpfulness", label="Helpful", logits={"Helpful": 0.1, "Neutral": 0.8, "Not helpful": 0.1}
        )

        # Add message evaluations to the evaluator
        self.evaluator.add_message_evaluation(
            turn="Hello, how can I help you?",
            author="bot",
            evaluation_results=[prediction1, prediction2, prediction3],
            seed_results={"seed_total": 0.8, "seed_intent": 0.8, "seed_output": 0.2, "seed_helpful": 0.1},
        )
        self.evaluator.add_message_evaluation(turn="I need help with Python.", author="user")

        # Get the DataFrame to be tested
        self.test_conversation_df = self.evaluator.get_evaluation_dataframe_with_bad_responses()

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

        save_conversation_with_evaluation(self.test_conversation_df, self.conv_id)

        # Check that the file was created
        self.assertTrue(os.path.exists(test_filename))

    def test_save_conversation_writes_correct_data(self):
        """Test that the function writes the correct data to the CSV file."""
        # Call function with a specific filename in the test directory
        test_filename = os.path.join(self.log_path, f"conversation_{self.conv_id}.csv")
        save_conversation_with_evaluation(self.test_conversation_df, self.conv_id)

        # Read the CSV file using pandas
        df = pd.read_csv(test_filename, sep=";")

        # Check header row (order doesn't matter)
        expected_columns = [
            "conv_id",
            "ID",
            "Author",
            "Message",
            "Evaluation",
            "Seed",
            "Reason for refinement",
        ]
        self.assertCountEqual(expected_columns, df.columns.tolist())

        # Check data rows
        self.assertEqual(len(df), 2)  # 2 messages

        # Check first message row
        first_row = df.iloc[0]
        self.assertEqual(first_row["conv_id"], int(self.conv_id))
        self.assertEqual(first_row["ID"], 1)
        self.assertEqual(first_row["Author"], "bot")
        self.assertEqual(first_row["Message"], "Hello, how can I help you?")

    def test_save_conversation_with_default_filename(self):
        """Test that the function creates a file with default filename when none is provided."""
        conv_id = datetime.now().strftime("%Y%m%d%H%M%S")
        expected_file_pattern = f"conversation_{conv_id}.csv"

        # mock the open function to avoid actual file creation
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            save_conversation_with_evaluation(self.test_conversation_df)

        # Check that open was called with the correct path
        # Extract the file path from the call arguments
        called_path = mock_file.call_args[0][0]
        self.assertTrue(called_path.endswith(expected_file_pattern))
        self.assertIn("logs", str(called_path))


if __name__ == "__main__":
    unittest.main()