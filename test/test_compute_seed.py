import unittest

from compute_seed import Seed


class TestSeed(unittest.TestCase):
    def setUp(self):
        # Default Seed instance for testing
        self.seed = Seed()
        # Custom weighted Seed instance for testing
        self.custom_seed = Seed(
            weight_intent=0.5, weight_output_elicitation=0.7, weight_helpfulness=0.3, weight_eliciting=0.8
        )

    def test_initialization(self):
        """Test that Seed initializes with correct default and custom weights"""
        # Test default weights
        self.assertEqual(self.seed.weight_intent, 1)
        self.assertEqual(self.seed.weight_output_elicitation, 1)
        self.assertEqual(self.seed.weight_helpfulness, 1)
        self.assertEqual(self.seed.weight_eliciting, 1)
        self.assertEqual(self.seed.l_min, 3)
        self.assertEqual(self.seed.l_max, 8)

        # Test custom weights
        self.assertEqual(self.custom_seed.weight_intent, 0.5)
        self.assertEqual(self.custom_seed.weight_output_elicitation, 0.7)
        self.assertEqual(self.custom_seed.weight_helpfulness, 0.3)
        self.assertEqual(self.custom_seed.weight_eliciting, 0.8)

    def test_compute_subscore_intent(self):
        """Test the intent subscore calculation"""
        # Test with no intent labels (D or I)
        no_intents = ["O", "O", "O", "O"]
        self.assertEqual(self.seed.compute_subscore_intent(no_intents), 0)

        # Test with all intent labels
        all_intents = ["D", "I", "D", "I"]
        self.assertEqual(self.seed.compute_subscore_intent(all_intents), 1)

        # Test with mixed labels
        mixed_intents = ["D", "O", "I", "O"]
        self.assertEqual(self.seed.compute_subscore_intent(mixed_intents), 0.5)

        # Test with empty list (edge case)
        self.assertEqual(self.seed.compute_subscore_intent([]), 0)

    def test_compute_subscore_output_elicitation(self):
        """Test the output elicitation subscore calculation"""
        # Test with short user replies (below l_min)
        short_replies = ["Hi", "OK", "Yes"]
        no_elicitation = ["No", "No", "No"]

        expected_score = 0  # LC=0 because ARL < l_min, ER=0 because no "Yes"
        actual_score = self.seed.compute_subscore_output_elicitation(short_replies, no_elicitation)
        self.assertEqual(actual_score, expected_score)

        # Test with long user replies (above l_min) and some elicitation
        long_replies = [
            "This is a long reply with many words",
            "Another fairly long reply from the user",
            "Third reply with sufficient word count",
        ]
        some_elicitation = ["Yes", "No", "Yes"]

        # Calculate expected LC: min((ARL/l_max), 1) where ARL > l_min
        ARL = (8 + 7 + 6) / 3
        LC = min((ARL/8), 1)
        ER = 2/3
        # Expected = 0.5 * (ER * weight_eliciting + LC)

        expected_score = round(0.5 *(ER + LC),2) # because weight_eliciting = 1
        actual_score = self.seed.compute_subscore_output_elicitation(long_replies, some_elicitation)
        self.assertEqual(actual_score, expected_score)


        # Test with empty lists (edge case)
        self.assertEqual(self.seed.compute_subscore_output_elicitation([], []), 0)

    def test_compute_subscore_helpfulness(self):
        """Test the helpfulness subscore calculation"""
        # Test with no helpful labels
        no_helpful = ["Neutral", "Not helpful", "Neutral"] # (n_neutral * 0.5) / total
        self.assertEqual(self.seed.compute_subscore_helpfulness(no_helpful), 0.33)

        # Test with all helpful labels
        all_helpful = ["Helpful", "Helpful", "Helpful"]
        self.assertEqual(self.seed.compute_subscore_helpfulness(all_helpful), 1)

        # Test with mixed labels
        mixed_helpful = ["Helpful", "Neutral", "Helpful", "Not helpful"]
        self.assertEqual(self.seed.compute_subscore_helpfulness(mixed_helpful), 0.62)

        # Test with empty list (edge case)
        self.assertEqual(self.seed.compute_subscore_helpfulness([]), 0)

    def test_compute_total_seed(self):
        """Test the total SEED score calculation"""
        # Sample user replies for all tests
        user_replies = ["i am ok", "ok", "who are you"]
        
        # Test with all perfect scores
        intent_labels_perfect = ["D", "I", "D"]  # All intents are D or I
        output_labels_perfect = ["Yes", "Yes", "Yes"]  # All elicitations are Yes
        helpfulness_labels_perfect = ["Helpful", "Helpful", "Helpful"]  # All helpful
        
        total_perfect = self.seed.compute_total_seed(intent_labels_perfect, output_labels_perfect,
                                                    helpfulness_labels_perfect, user_replies)["seed"]

        # With perfect labels, this should be close to 1
        self.assertAlmostEqual(total_perfect, 1.0, places=1)
        
        # Test with all zero scores
        intent_labels_zero = ["O", "O", "O"]  # No intents
        output_labels_zero = ["No", "No", "No"]  # No elicitations
        helpfulness_labels_zero = ["Not helpful", "Not helpful", "Not helpful"]  # None helpful
        
        total_zero = self.seed.compute_total_seed(intent_labels_zero, output_labels_zero, 
                                                 helpfulness_labels_zero, user_replies)["seed"]
        self.assertAlmostEqual(total_zero, 0.0, places=1)
        
        # Test with mixed scores
        intent_labels_mixed = ["D", "O", "O"]  # 1/3 intents
        output_labels_mixed = ["Yes", "No", "No"]  # 1/3 elicitations
        helpfulness_labels_mixed = ["Helpful", "Neutral", "Not helpful"]  # Mixed helpfulness
        
        total_mixed = self.seed.compute_total_seed(intent_labels_mixed, output_labels_mixed, 
                                                  helpfulness_labels_mixed, user_replies)["seed"]
        # The exact value will depend on implementation details but should be in mid-range
        self.assertTrue(0.2 <= total_mixed <= 0.6)
        
        # Test with custom weights
        total_custom = self.custom_seed.compute_total_seed(intent_labels_mixed, output_labels_mixed, 
                                                         helpfulness_labels_mixed, user_replies)["seed"]
        # Should be different from total_mixed due to custom weights
        self.assertNotEqual(total_custom, total_mixed)

    def test_end_to_end_computation(self):
        """Test an end-to-end SEED calculation with sample data"""
        # Sample data
        intent_labels = ["D", "O", "I", "O", "D"]
        user_replies = [
            "Hello there",
            "I'm interested in learning more about this topic",
            "Can you explain how this works?",
            "That makes sense, thank you for explaining",
        ]
        elicitation_labels = ["No", "Yes", "No", "Yes"]
        helpfulness_labels = ["Helpful", "Neutral", "Helpful", "Helpful", "Not helpful"]

        # Calculate total SEED score
        intent_subscore = self.seed.compute_subscore_intent(intent_labels)
        output_subscore = self.seed.compute_subscore_output_elicitation(user_replies, elicitation_labels)
        helpfulness_subscore = self.seed.compute_subscore_helpfulness(helpfulness_labels)
        total_seed = round(self.seed.compute_total_seed(intent_labels, elicitation_labels, helpfulness_labels, user_replies)["seed"], 2)

        # Manually compute expected values
        expected_intent = 3 / 5  # 3 D/I labels out of 5

        # For output elicitation:
        ARL = (2 + 9 + 6 + 7) / 4
        LC = min((ARL/8), 1)
        ER = 2/4
        # Expected = 0.5 * (ER * weight_eliciting + LC)

        expected_output = round(0.5 * (ER + LC),2) # because weight_eliciting = 1
        expected_helpfulness = round(3.5 / 5, 2)  # 3 Helpful labels out of 5

        # Expected total with default weights (all 1)
        expected_total = round((expected_intent + expected_output + expected_helpfulness)/3,2)

        # Verify calculations
        self.assertAlmostEqual(intent_subscore, expected_intent)
        self.assertAlmostEqual(output_subscore, expected_output)
        self.assertAlmostEqual(helpfulness_subscore, expected_helpfulness)
        self.assertAlmostEqual(total_seed, expected_total)


if __name__ == "__main__":
    unittest.main()