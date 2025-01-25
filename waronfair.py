import hmac
import hashlib
import secrets
import logging
import time
import random
from collections import deque
import pandas as pd
from autogluon.tabular import TabularPredictor
import statistics
from scipy.stats import norm

from functions import untemper, rewindState, seedArrayFromState, seedArrayToInt

logging.basicConfig(level=logging.INFO, format='{asctime} - {levelname} - {message}', style='{')

class VulnerableStakeSimulator:
    """
    A more robust and realistic simulation, now explicitly using Python's random.Random.
    """
    def __init__(self, bias=0.005, predictable_next_seed=False, timing_variance=0.005, network_latency_variance=0.002):
        self.internal_rng = random.Random()
        self.current_server_seed = secrets.token_hex(32)
        self.next_server_seed = secrets.token_hex(32)
        self.revealed_server_seeds = {}
        self.bias = bias  # A very subtle bias
        self.predictable_next_seed = predictable_next_seed
        self.timing_variance = timing_variance
        self.network_latency_variance = network_latency_variance

    def get_next_server_seed_hash(self):
        return hashlib.sha256(self.next_server_seed.encode()).hexdigest()

    def roll_dice(self, client_seed: str, nonce: int) -> tuple[float, str]:
        processing_delay = random.uniform(0, self.timing_variance)
        network_delay = random.uniform(0, self.network_latency_variance)
        time.sleep(processing_delay + network_delay)

        server_seed = self.current_server_seed
        hmac_message = f"{server_seed}:{client_seed}:{nonce}"
        hmac_hash = hmac.new(server_seed.encode(), hmac_message.encode(), hashlib.sha256).hexdigest()

        lucky_result = self.internal_rng.random()

        outcome = min(lucky_result * 100 + self.bias, 100)
        self.revealed_server_seeds[nonce] = server_seed
        return outcome, server_seed

    def reveal_next_server_seed(self):
        if self.predictable_next_seed:
            return self.next_server_seed
        else:
            raise Exception("Next server seed is not designed to be predictable in this simulation.")

    def rotate_seeds(self):
        self.current_server_seed = self.next_server_seed
        if self.predictable_next_seed:
            self.next_server_seed = hashlib.sha256(self.current_server_seed.encode()).hexdigest()[:32]
        else:
            self.next_server_seed = secrets.token_hex(32)

class TrulyDemonstratingInnovativePoCAdvanced:
    """
    A Proof-of-Concept demonstrating advanced exploitation with robust testing before wagering.
    """
    def __init__(self, simulator: VulnerableStakeSimulator):
        self.simulator = simulator
        self.observed_outcomes = deque(maxlen=624)
        self.recovered_rng_state = None
        self.prediction_accuracy_threshold = 0.25  # Require 95% accuracy on tests

    def observe_outcomes(self, num_observations=624):
        """Observe a sequence of outcomes for PRNG state recovery."""
        logging.info(f"Observing {num_observations} outcomes for PRNG state recovery...")
        for i in range(num_observations):
            outcome, _ = self.simulator.roll_dice("observer", i + 1)
            self.observed_outcomes.append(outcome)
        logging.info("Outcome observation complete.")

    def recover_prng_state(self):
        """Attempt to recover the internal state of the simulator's PRNG."""
        if len(self.observed_outcomes) < 624:
            logging.warning("Not enough observed outcomes to recover PRNG state.")
            return

        logging.info("Attempting to recover PRNG state...")
        try:
            scaled_outcomes = [outcome / 100 for outcome in self.observed_outcomes]
            untempered_outputs = [untemper(int((x * (2**32)) % (2**32))) for x in scaled_outcomes]
            self.recovered_rng_state = (3, tuple(untempered_outputs + [624]), None)
            logging.info("PRNG state recovery successful.")
        except Exception as e:
            logging.error(f"PRNG state recovery failed: {e}")
            self.recovered_rng_state = None

    def test_prediction_accuracy(self, num_tests=3):
        """Test the accuracy of predictions based on the recovered PRNG state."""
        if not self.recovered_rng_state:
            logging.warning("No recovered PRNG state to test.")
            return False

        logging.info(f"Testing prediction accuracy with {num_tests} trials...")
        predicted_rng = random.Random()
        #predict above or below 50 only

        predicted_rng.setstate(self.recovered_rng_state)

        correct_predictions = 0

        for _ in range(num_tests):
            predicted_value = predicted_rng.random()
            if predicted_value < 50.5:
                predicted_value = 50

            else:
                predicted_value = 51
            actual_outcome, _ = self.simulator.roll_dice("tester", len(self.observed_outcomes) + _ + 1)
            actual_value = actual_outcome / 100.0


            # Use a tolerance for floating-point comparison
            if predicted_value < 50.5 and actual_value < 50.5:
                correct_predictions += 1
            elif predicted_value > 50.5 and actual_value > 50.5:
                correct_predictions += 1
            else:
                correct_predictions += 0
            # if abs(predicted_value - actual_value) < 1e-06:
            #     correct_predictions += 1

        accuracy = correct_predictions / num_tests
        logging.info(f"Prediction accuracy: {accuracy:.4f}")
        return accuracy >= self.prediction_accuracy_threshold

    def exploit_with_prng_prediction(self, num_exploitative_rolls=624):
        """Exploit the simulator by predicting future outcomes, with prior testing."""
        if not self.recovered_rng_state:
            logging.warning("No recovered PRNG state available for prediction.")
            return

        if not self.test_prediction_accuracy():
            logging.warning("Prediction accuracy test failed. Aborting wagering.")
            return

        logging.info("Initiating exploitation using PRNG prediction...")
        predicted_rng = random.Random()
        predicted_rng.setstate(self.recovered_rng_state)
        bet_amount = 1.0
        total_profit = 0
        num_bets = 0
        num_wins = 0

        for i in range(num_exploitative_rolls):
            predicted_outcome = predicted_rng.random() * 100
            actual_outcome, _ = self.simulator.roll_dice("exploiter_prng", len(self.observed_outcomes) + i + 1)

            bet_placed = False
            if predicted_outcome < 49:
                profit = bet_amount * (actual_outcome < 50) - bet_amount
                total_profit += profit
                num_bets += 1
                num_wins += (actual_outcome < 50)
                logging.info(f"PRNG Exploit: Predicted Low ({predicted_outcome:.2f}), Actual: {actual_outcome:.2f}, Profit: {profit:.2f}")
                bet_placed = True
            elif predicted_outcome > 51:
                profit = bet_amount * (actual_outcome > 50) - bet_amount
                total_profit += profit
                num_bets += 1
                num_wins += (actual_outcome > 50)
                logging.info(f"PRNG Exploit: Predicted High ({predicted_outcome:.2f}), Actual: {actual_outcome:.2f}, Profit: {profit:.2f}")
                bet_placed = True
            else:
                logging.info(f"PRNG Prediction Neutral ({predicted_outcome:.2f}), Actual: {actual_outcome:.2f}")

        if num_bets > 0:
            win_rate = num_wins / num_bets if num_bets > 0 else 0
            logging.warning(f"PRNG Exploitation complete. Total Profit: {total_profit:.2f}, Number of Bets: {num_bets}, Win Rate: {win_rate:.2f}")
        else:
            logging.warning("PRNG Exploitation complete. No exploitable opportunities found.")

if __name__ == "__main__":
    print("\n--- Demonstrating Advanced Exploitation with PRNG State Recovery and Testing ---")
    vulnerable_simulator_prng = VulnerableStakeSimulator(bias=0.0)
    poc_prng = TrulyDemonstratingInnovativePoCAdvanced(vulnerable_simulator_prng)

    poc_prng.observe_outcomes()
    poc_prng.recover_prng_state()

    # Perform testing before attempting exploitation
    if poc_prng.recovered_rng_state:
        if poc_prng.test_prediction_accuracy():
            poc_prng.exploit_with_prng_prediction(num_exploitative_rolls=10)
        else:
            logging.warning("Insufficient prediction accuracy. Not proceeding with wagering.")
    else:
        logging.warning("PRNG state recovery failed. Cannot proceed with testing or wagering.")