import numpy as np


class QLearning:
    def __init__(self, num_phase, max_num_car_stopped, num_lane, num_action):
        self.q_table = np.random.uniform(low=-1, high=1, size=(num_phase*max_num_car_stopped**num_lane, num_action))
        self.episode = 0
        self.epsilon = 0.5 * (1 / (self.episode + 1))
        self.action = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
        self.is_set_duration = False
        self.is_calculate_next_action = False
        self.previous_action = None
        self.previous_digitized_state = None
        self.max_num_car_stopped = max_num_car_stopped
        self.next_action_idx = 0

    def digitize_state(self, state_dict):
        light_phase = state_dict['light_phase']
        nums_car_stopped = state_dict['nums_car_stopped']
        digitized = [
            nums_car_stopped[0],
            nums_car_stopped[1],
            nums_car_stopped[2],
            nums_car_stopped[3]
        ]

        # TODO: わかりづらいのであとでなおす
        return sum([x * (self.max_num_car_stopped**i) for i, x in enumerate(digitized)]) + (int(light_phase/2) * 10000)

    def get_action(self, next_state):
        # ε-greedy
        # 1000stepごとにεを減らす
        self.episode += 1
        decrease_param = 1 / (np.ceil(self.epsilon / 1000) + 1)
        epsilon = 0.5 * decrease_param

        if epsilon <= np.random.uniform(0, 1):
            next_action_idx = np.argmax(self.q_table[next_state])
        else:
            next_action_idx = np.random.choice(10)
        return next_action_idx

    def calculate_reward(self):
        pass

    def update_Qtable(self, state, action, reward, next_state):
        gamma = 0.99
        alpha = 0.5

        next_max_Q = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (reward + gamma * next_max_Q)
