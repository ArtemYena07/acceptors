import itertools
import random
from typing import Optional
from utils import make_tree_like_acceptor, hopkroft_minimize_acceptor
from q_learn_small_states import is_finish
from acceptor import Acceptor
import pickle
import numpy as np
from itertools import product


def count_synthesis(allowed_words: list, banned_words: list, alphabet: set,
                    acceptor: Optional[Acceptor] = None) -> None:
  if not acceptor:
      acceptor = make_tree_like_acceptor(allowed_words, alphabet)
      acceptor = hopkroft_minimize_acceptor(acceptor)
  errors, succ = 0, 0
  print(len(acceptor.states))
  changed = True
  while changed:
    changed = False
    states = list(acceptor.states.keys())
    for state, letter in itertools.product(list(filter(lambda x: x != 'trash', states)), alphabet):
      if acceptor.get_new_state(state, letter) == 'trash':
        new_state = random.choice(list(filter(lambda x: x != 'trash' and x != state, states)))
        acceptor.set_transition(state, letter, new_state)
        new_acceptor = hopkroft_minimize_acceptor(acceptor)
        if any(map(lambda word: new_acceptor.accept_word(word), banned_words)):
          acceptor.set_transition(state, letter, 'trash')
          errors += 1
        else:
          acceptor = new_acceptor
          succ += 1
          changed = True
          break
  print(errors, succ, len(acceptor.states))


def count_learned(table, states_dict, states_actions_dict,
                  acceptable_words, banned_words, alphabet):
    acceptor = make_tree_like_acceptor(acceptable_words, alphabet)
    acceptor = hopkroft_minimize_acceptor(acceptor)
    errors, succ = 0, 0
    done = False
    while not done:
        cnt = 0
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
            errors += 1
            print(errors, succ, len(acceptor.states))
            print('reached wrong state')
            count_synthesis(acceptable_words, banned_words, alphabet, acceptor)
            break
        state_num = states_dict[state_array]
        action = np.argmax(table[state_num])
        state, new_state, letter = states_actions_dict[state_num][action]
        acceptor.set_transition(state, letter, new_state)
        new_acceptor = hopkroft_minimize_acceptor(acceptor)
        if any(map(lambda word: new_acceptor.accept_word(word), banned_words)):
            print('wrong')
            acceptor.set_transition(state, letter, 'trash')
            errors += 1
            states = list(acceptor.states.keys())
            for state, letter in itertools.product(list(filter(lambda x: x != 'trash', states)), alphabet):
                if acceptor.get_new_state(state, letter) == 'trash':
                    new_state = random.choice(list(filter(lambda x: x != 'trash' and x != state, states)))
                    acceptor.set_transition(state, letter, new_state)
                    new_acceptor = hopkroft_minimize_acceptor(acceptor)
                    if any(map(lambda word: new_acceptor.accept_word(word), banned_words)):
                        acceptor.set_transition(state, letter, 'trash')
                        errors += 1
                    else:
                        acceptor = new_acceptor
                        succ += 1
                        break
        else:
            acceptor = new_acceptor
            succ += 1

        done = is_finish(acceptor, banned_words)
    print(errors, succ, len(acceptor.states))


if __name__ == '__main__':
  with open('data.pickle', 'rb') as f:
    table, states_dict, states_actions_dict = pickle.load(f)

  count_synthesis(
        ['abbab', 'cab', 'baaab'],
        ['ababba', 'ccc', 'acab'],
        {'a', 'b', 'c'}
  )

  count_learned(table, states_dict, states_actions_dict,
                ['abbab', 'cab', 'baaab'],
                ['ababba', 'ccc', 'acab'],
                {'a', 'b', 'c'}
  )
