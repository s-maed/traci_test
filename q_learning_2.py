import numpy as np


class QLearning:
    def __init__(self, phases, num_lane_occupancy_states, num_lanes, min_elapsed_time, max_elapsed_time, actions):

        self.q_table = np.random.uniform(low=0, high=1, size=(len(phases) * num_lane_occupancy_states**num_lanes * (max_elapsed_time - min_elapsed_time), len(actions)))
        self.phases = phases
        self.num_lane_occupancy_states = num_lane_occupancy_states
        self.num_lanes = num_lanes
        self.min_elapsed_time = min_elapsed_time
        self.max_elapsed_time = max_elapsed_time
        self.actions = actions

        self.is_set_max_duration = False
        self.prev_t = 0
        self.action = 0
        self.state = 0

        self.max_length_prev_t = 0
        self.rewards = []
        self.cycle_rewards = 0



    def digitize_state(self, light_phase, ns_occupancy, ew_occupancy, elapsed_time):

        digitized = int(light_phase/2)
        digitized += len(self.phases) * np.digitize(ns_occupancy, bins=bins(0, 0.9, 10))
        digitized += len(self.phases) * self.num_lane_occupancy_states * np.digitize(ew_occupancy, bins=bins(0, 0.9, 10))
        digitized += len(self.phases) * self.num_lane_occupancy_states**2 * elapsed_time

        return digitized


    def get_action(self, observation):
        # ε-greedy, 1000stepごとにεを減らす
        decrease_param = 1 / (np.ceil(self.prev_t / 20000) + 1)
        epsilon = 0.5 * decrease_param

        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[observation])
        else:
            next_action = np.random.choice(self.actions)
        return next_action

    def calculate_reward(self, ns_length, ew_lenght):
        max_length_t = ns_length + ew_lenght
        reward = self.max_length_prev_t**2 - max_length_t**2
        return reward

    def update_Qtable(self, state, action, reward, observation):
        gamma = 0.5
        alpha = 0.5

        next_max_Q = np.max(self.q_table[observation])
        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (reward + gamma * next_max_Q)


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

