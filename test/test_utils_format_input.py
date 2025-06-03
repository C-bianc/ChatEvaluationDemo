import unittest
from unittest.mock import Mock

from app.utils.model_input_formatting import format_input


class TestFormatInput(unittest.TestCase):

    def test_format_input_single_user_message(self):
        tokenizer = Mock()
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.bos_token = "[SEP]"

        convo = [{"content": "Hello"}]
        target = "How are you?"
        expected_output = "[CLS]Hello[SEP]How are you?[SEP]"

        result = format_input(convo, target, tokenizer)
        self.assertEqual(result, expected_output)

    def test_format_input_no_prior_context(self):
        tokenizer = Mock()
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.bos_token = "[SEP]"

        convo = []
        target = "Let's get started"
        expected_output = "[CLS]Let's get started[SEP]"

        result = format_input(convo, target, tokenizer)
        self.assertEqual(result, expected_output)

    def test_format_input_with_multiple_turns(self):
        tokenizer = Mock()
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.bos_token = "[SEP]"

        convo = [
            {"content": "Hi"},
            {"content": "Hello! How can I help you?"},
            {"content": "I need assistance with my account"},
        ]
        target = "Sure, happy to help."

        expected_output = "[CLS]Hello! How can I help you?[SEP]I need assistance with my account[SEP]Sure, happy to help.[SEP]"

        result = format_input(convo, target, tokenizer)
        self.assertEqual(result, expected_output)

    def test_format_input_with_empty_target(self):
        tokenizer = Mock()
        tokenizer.cls_token = "[CLS]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.bos_token = "[SEP]"

        convo = []
        target = "Hi"
        expected_output = "[CLS]Hi[SEP]"

        result = format_input(convo, target, tokenizer)
        self.assertEqual(result, expected_output)



if __name__ == "__main__":
    unittest.main()
