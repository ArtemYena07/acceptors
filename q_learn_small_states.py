import utils
from itertools import product
import numpy as np
import random
from acceptor import Acceptor
from typing import List, Tuple


def is_finish(acceptor: Acceptor, banned_words: List[str]) -> bool:
    states = list(acceptor.states.keys())
    alphabet = acceptor.alphabet
    for state, letter in product(list(filter(lambda x: x != 'trash', states)), alphabet):
        if acceptor.get_new_state(state, letter) == 'trash':
            for new_state in filter(lambda x: x != 'trash' and x != state, states):
                acceptor.set_transition(state, letter, new_state)
                if any(map(lambda word: acceptor.accept_word(word), banned_words)):
                    acceptor.set_transition(state, letter, 'trash')
                else:
                    acceptor.set_transition(state, letter, 'trash')
                    return False
    return True


def process_step_v1(acceptor: Acceptor, from_state_name: str,
                    to_state_name: str, letter: str, banned_words: List[str]
                    ) -> Tuple[int, Acceptor, bool]:
    current_destination = acceptor.get_new_state(from_state_name, letter)
    if current_destination != 'trash':
        return -10, acceptor, False

    acceptor.set_transition(from_state_name, letter, to_state_name)

    previous_num_of_states = len(acceptor.states)

    if any(map(lambda word: acceptor.accept_word(word), banned_words)):
        acceptor.set_transition(from_state_name, letter, 'trash')
        return -5, acceptor, False
    else:
        acceptor = utils.hopkroft_minimize_acceptor(acceptor)
        new_num_of_states = len(acceptor.states)
        return 1 + 15 * (previous_num_of_states - new_num_of_states), \
               utils.hopkroft_minimize_acceptor(acceptor), is_finish(acceptor, banned_words)


def generate_q_table(acceptable_words, banned_words, alphabet,
                     alpha=0.1, gamma=0.47, epsilon=0.5, n_repeats=20000):

    acceptor = utils.make_tree_like_acceptor(acceptable_words, alphabet)
    acceptor = utils.hopkroft_minimize_acceptor(acceptor)

    states_dict = {}
    state_array = np.array(list(acceptor.states.keys()))

    q_table = {0: np.zeros(acceptor.actions_num)}
    states_cnt = 0
    state_array.sort()
    states_dict[tuple(state_array)] = 0

    actions_dict = {}
    actions_count = 0
    for state1 in sorted(acceptor.states.keys()):
        for state2 in sorted(acceptor.states.keys()):
            if state1 == 'trash' or state2 == 'trash':
                continue
            for letter in sorted(alphabet):
                actions_dict[actions_count] = (state1, state2, letter)
                actions_count += 1

    states_qty = []
    steps = []
    states_actions_dict = {0: actions_dict}

    for i in range(0, n_repeats):
        state = 0

        epochs, penalties, reward, = 0, 0, 0
        done = False
        acceptor = utils.make_tree_like_acceptor(acceptable_words, alphabet)
        acceptor = utils.hopkroft_minimize_acceptor(acceptor)

        while not done:
            actions_dict = states_actions_dict[state]

            if random.uniform(0, 1) < 1 - (i / n_repeats):
                action = np.random.randint(0, len(actions_dict))  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values
            # print(action, q_table[state][action])

            state_from, state_to, letter = actions_dict[action]

            reward, acceptor, done = process_step_v1(
                acceptor, state_from, state_to, letter, banned_words
            )

            state_array = np.array(list(acceptor.states.keys()))
            state_array.sort()

            state_array = tuple(state_array)

            if state_array not in states_dict:
                states_cnt += 1
                states_dict[state_array] = states_cnt
                q_table[states_cnt] = np.zeros(acceptor.actions_num)
                actions_dict = {}
                actions_count = 0
                for state1 in sorted(acceptor.states.keys()):
                    for state2 in sorted(acceptor.states.keys()):
                        if state1 == 'trash' or state2 == 'trash':
                            continue
                        for letter in sorted(alphabet):
                            actions_dict[actions_count] = (state1, state2, letter)
                            actions_count += 1

                states_actions_dict[states_cnt] = actions_dict

            next_state = states_dict[state_array]

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            #print(reward, next_state, new_value)
            q_table[state][action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        steps.append(i)
        states_qty.append(len(q_table))
        if i % 500 == 0:
            print(f"Episode: {i}")

    print("Training finished.\n")
    return q_table, steps, states_qty, states_dict, states_actions_dict
