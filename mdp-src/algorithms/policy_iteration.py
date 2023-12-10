from utilities.utilities import *


def evaluate_policy(policy: List[List[str]],
                    rewards: List[List[int]],
                    gamma: float) -> List[List[float]]:
    """

    :param policy:
    :param rewards:
    :param gamma:
    :return:
    """

    last_value = []
    current_value = [[0 for _ in range(len(rewards[0]))] for _ in range(len(rewards))]

    while not check_value_convergence(current_value, last_value):
        last_value = current_value

        # update current_value
        current_value = compute_new_value(current_value, rewards, policy, gamma)

    return current_value


def extract_policy(rewards: List[List[int]],
                   policy_values: List[List[float]],
                   gamma: float) -> List[List[str]]:
    pass


def policy_iteration(rewards: List[List[int]],
                     initial_policy: List[List[str]],
                     gamma: float,
                     epsilon: float = 1e-7):
    """
    Function implements the policy iteration algorithm, using the methods evaluate_policy and extract_policy.

    :param rewards: Represents an array of rewards of the game board.
    :param initial_policy:
    :param gamma: Discount factor
    :param epsilon:
    :return:
    """

    last_policy: List[List[str]] = [[]]
    current_policy: List[List[str]] = initial_policy

    while not check_policy_convergence(current_policy, last_policy):
        last_policy = current_policy

        # Step 1 - Policy Evaluation
        policy_values = evaluate_policy(current_policy, rewards, gamma)

        # Step 2 - Policy Extraction
        current_policy = extract_policy(rewards, policy_values, gamma)



