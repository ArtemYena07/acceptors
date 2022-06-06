import utils
from itertools import product
import numpy as np
import random
from acceptor import Acceptor
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt


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


def process_step_v2(acceptor: Acceptor, from_state_name: str,
                    to_state_name: str, letter: str, banned_words: List[str]) -> Tuple[int, Acceptor, bool]:
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
        return 15 * (previous_num_of_states - new_num_of_states), \
               utils.hopkroft_minimize_acceptor(acceptor), is_finish(acceptor, banned_words)


def generate_q_table(acceptable_words, banned_words, alphabet,
                     alpha=0.2, gamma=0.6, epsilon=0.5, n_repeats=100000):

    acceptor = utils.make_tree_like_acceptor(acceptable_words, alphabet)
    acceptor = utils.hopkroft_minimize_acceptor(acceptor)

    states_dict = {}
    state_letter_prod = list(
        product(list(filter(lambda x: x != 'trash', list(acceptor.states.keys()))), alphabet)
    )

    cnt = 0

    max_state_name_len = max(map(len, acceptor.states.keys()))
    state_array = np.array(['a' * (max_state_name_len * 3 + 2) for _ in state_letter_prod])
    for state, letter in state_letter_prod:
        state_array[cnt] = state + '_' + letter + '_' + acceptor.get_new_state(state, letter)
        cnt += 1

    q_table = {0: np.zeros(acceptor.actions_num)}
    states_cnt = 0
    state_array.sort()
    states_dict[tuple(state_array)] = 0

    base_state_array = tuple(state_array)

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
    epoch = 0

    while epoch < n_repeats:
        state = 0

        penalties, reward, = 0, 0
        done = False
        state_array = np.array(base_state_array)
        acceptor = utils.make_tree_like_acceptor(acceptable_words, alphabet)
        acceptor = utils.hopkroft_minimize_acceptor(acceptor)

        while not done:
            eps_new = epsilon - (epsilon * epoch / n_repeats)
            actions_dict = states_actions_dict[state]

            if random.uniform(0, 1) < eps_new:
                action = np.random.randint(0, acceptor.actions_num)  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            state_from, state_to, letter = actions_dict[action]

            reward, acceptor, done = process_step_v2(
                acceptor, state_from, state_to, letter, banned_words
            )

            state_letter_prod = list(
                product(list(filter(lambda x: x != 'trash', list(acceptor.states.keys()))), alphabet))
            cnt = 0

            max_state_name_len = max(map(len, acceptor.states.keys()))
            state_array = np.array(['a' * (max_state_name_len * 3 + 2) for _ in state_letter_prod])
            for state_name, letter in state_letter_prod:
                state_array[cnt] = state_name + '_' + letter + '_' + acceptor.get_new_state(state_name, letter)
                cnt += 1

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

            # if next_state == state:
            #   print(state_array)

            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state][action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epoch += 1
            steps.append(epoch)
            states_qty.append(len(q_table))

            if epoch % 50000 == 0:
                print(f"Episode: {epoch}")

            if epoch == n_repeats:
                break

    print("Training finished.\n")
    return q_table, steps, states_qty, states_dict, states_actions_dict


if __name__ == '__main__':
    table, steps, states_qty, states_dict, states_actions_dict = generate_q_table(
        # ['cbacb', 'abcbab', 'babcbabc'],
        # ['bcbacbacb', 'ba'],
        # ['acb', 'abac', 'cb', 'bb'],
        # ['bcabcba', 'bcbcabca'],
         ['bac', 'cacc', 'cab'],

        # ['abcbca', 'bcabc', 'b'],
        # ['a', 'b', 'c'],
        # ['aa', 'bc'],
        # ['bcab', 'acab', 'bac'],
         ['aaaa', 'ccc'],
        {'a', 'b', 'c'},
        n_repeats=200000, epsilon=1
    )
    with open('data.pickle', 'wb') as f:
        pickle.dump((table, states_dict, states_actions_dict), f)

    # print(table)
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    ax.plot(steps, states_qty)
    ax.set_xlabel(r"$t_{i}$", fontsize=12)
    ax.set_ylabel(r"$|S|$", fontsize=12, rotation=0)
    ax.set_title(r"$|S|$ for every $t_{i}$", fontsize=14)

    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    n, bins, _ = ax.hist(list(map(lambda x: int(np.sqrt(len(x) / 3) + 1), table.values())))
    ax.set_xlabel(r"$|Q_{s}|$", fontsize=12)
    ax.set_ylabel(r"Number of such $s$ in $S$", fontsize=12)
    ax.set_title(r"Number of configurations with such number of states", fontsize=14)

    plt.show()