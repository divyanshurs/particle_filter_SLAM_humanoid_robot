class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """

        self._game = game
        self.__discount = discount
        self._values = {}

    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        return self._values.get(state, 0.)

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        trans = self._game.get_transitions(state, action)
        cur_sum = 0.
        for n_state, prob in trans.items():
            cur_sum += prob * (
                    self._game.get_reward(state, action, n_state)
                    + self.__discount * self.get_value(n_state))
        return cur_sum

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        max_arg = None
        max_value = float('-inf')
        for action in self._game.get_actions(state):
            v = self.get_q_value(state, action)
            if v > max_value:
                max_value = v
                max_arg = action
        return max_arg

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        new_values = {}
        for state in self._game.states:
            new_values[state] = max(
                (self.get_q_value(state, action)
                 for action in self._game.get_actions(state)),
                default=0.)
        self._values = new_values


class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function or
    override ValueIterationAgent's methods, you can add them as well.
    """

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until |V_{k+1}(s) - V_k(s)| < ε
        """
        epsilon = 1e-6

        policy = {}
        for state in self._game.states:
            policy[state] = self.get_best_policy(state)

        while True:
            max_diff = 0
            for state in self._game.states:
                before = self.get_value(state)
                after = self._values[state] = self.get_q_value(state, policy[state])
                max_diff = max(max_diff, abs(before - after))
            if max_diff < epsilon:
                break


def question_3():
    discount = 0.9
    noise = 0
    return discount, noise


def question_4a():
    discount = 0.2
    noise = 0
    living_reward = -3
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4b():
    discount = 0.5
    noise = 0.1
    living_reward = -1
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4c():
    discount = 0.9
    noise = 0
    living_reward = 0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4d():
    discount = 0.9
    noise = 0.1
    living_reward = 0.5
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4e():
    discount = 0.1
    noise = 0
    living_reward = 10
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'

feedback_question_1 = 2

feedback_question_2 = """lol"""

feedback_question_3 = """lol"""