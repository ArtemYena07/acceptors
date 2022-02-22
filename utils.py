from functools import reduce
from acceptor import Acceptor
from queue import Queue
from typing import Tuple
import random
import itertools


def make_tree_like_acceptor(words: list, alphabet: set = None):
    if not alphabet:
        alphabet = reduce(lambda acc, elem: set.union(acc, set(elem)), words, set())

    acceptor = Acceptor(alphabet)
    acceptor.set_start('')
    for word in words:
        cur = ''
        for letter in word:
            prev = cur
            cur = cur + letter
            acceptor.set_transition(prev, letter, cur)
        acceptor.set_finish(word)

    return acceptor


def split_cl(acceptor: Acceptor, R: Tuple[str], C: Tuple[str], a: str) -> Tuple[Tuple[str], Tuple[str]]:
    R1 = []
    R2 = []
    for state in R:
        if acceptor.get_new_state(state, a) in C:
            R1.append(state)
        else:
            R2.append(state)
    return tuple(R1), tuple(R2)


def minimize_acceptor(acceptor: Acceptor) -> Acceptor:
    P = {tuple(acceptor.finish), tuple(set(acceptor.states.keys()) - acceptor.finish)}
    S = Queue()

    alphabet = acceptor.alphabet
    for c in alphabet:
        S.push((acceptor.finish, c))
        S.push((set(acceptor.states.keys()) - acceptor.finish, c))

    while not S.is_empty():
        C, a = S.pop()
        buf = set()
        diff = set()
        for R in P:
            R1, R2 = split_cl(acceptor, R, C, a)
            if len(R1) and len(R2):
                diff.add(R)
                buf.add(R1)
                buf.add(R2)
                for c in alphabet:
                    S.push((R1, c))
                    S.push((R2, c))
        P = P - diff
        P = P | buf

    P_names = set()
    states_map = {}
    for i in P:
        name = reduce(lambda acc, elem: acc + elem, i, '')
        P_names.add(name)
        for state in i:
            states_map[state] = name

    new_acceptor = Acceptor(alphabet)

    for state in acceptor.states:
        for letter in alphabet:
            new_acceptor.set_transition(states_map[state], letter, states_map[acceptor.get_new_state(state, letter)])
            if state in acceptor.finish:
                new_acceptor.set_finish(states_map[state])
            if state == acceptor.start.name:
                new_acceptor.set_start(states_map[state])

    return new_acceptor


def synthesis_method(allowed_words: list, banned_words: list, alphabet: set) -> Acceptor:
  acceptor = make_tree_like_acceptor(allowed_words, alphabet)
  acceptor, _ = minimize_acceptor(acceptor)

  changed = True
  while changed:
    changed = False
    states = list(acceptor.states.keys())
    for state, letter in itertools.product(list(filter(lambda x: x != 'trash', states)), alphabet):
      if acceptor.get_new_state(state, letter) == 'trash':
        new_state = random.choice(list(filter(lambda x: x != 'trash' and x != state, states)))
        acceptor.set_transition(state, letter, new_state)
        new_acceptor, _ = minimize_acceptor(acceptor)
        if any(map(lambda word: new_acceptor.accept_word(word), banned_words)):
          acceptor.set_transition(state, letter, 'trash')
        else:
          acceptor = new_acceptor
          changed = True
          break
  return acceptor


def hopkroft_minimize_acceptor(acceptor: Acceptor) -> Acceptor:
    inv = {}
    alphabet = acceptor.alphabet

    for state_name in acceptor.states:
        for letter in alphabet:
            destination_name = acceptor.get_new_state(state_name, letter)
            if destination_name in inv:
                if letter in inv[destination_name]:
                    inv[destination_name][letter].append(state_name)
                else:
                    inv[destination_name][letter] = [state_name]
            else:
                inv[destination_name] = {letter: [state_name]}

            if state_name in inv:
                if letter not in inv[state_name]:
                    inv[state_name][letter] = []
            else:
                inv[state_name] = {letter: []}

    states_classes = {}
    P = {0: set(), 1: set()}

    for state_name in acceptor.finish:
        states_classes[state_name] = 0
        P[0].add(state_name)

    for state_name in (set(acceptor.states.keys()) - acceptor.finish):
        states_classes[state_name] = 1
        P[1].add(state_name)

    S = Queue()

    for c in alphabet:
        S.push((0, c))
        S.push((1, c))

    count = {}
    twin = {}

    while not S.is_empty():
        C, a = S.pop()
        involved = set()

        for q in P[C]:
            for r in inv[q][a]:
                i = states_classes[r]
                involved.add(i)
                if i in count:
                    count[i] += 1
                else:
                    count[i] = 1

        for i in involved:
            if count[i] < len(P[i]):
                index = len(P)
                P[index] = set()
                twin[i] = index

        for q in P[C].copy():
            for r in inv[q][a]:
                i = states_classes[r]
                j = twin.get(i, 0)
                if j != 0:
                    P[i].remove(r)
                    P[j].add(r)

        for i in involved:
            j = twin.get(i, 0)
            if j != 0:
                if len(P[j]) > len(P[i]):
                    P[i], P[j] = P[j], P[i]
                for r in P[j]:
                    states_classes[r] = j
                for c in alphabet:
                    S.push((j, c))
            count[i] = 0
            twin[i] = 0

    P_names = set()
    states_map = {}
    for i in P.values():
        name = reduce(lambda acc, elem: acc + elem, i, '')
        P_names.add(name)
        for state in i:
            states_map[state] = name

    new_acceptor = Acceptor(alphabet)

    for state in acceptor.states:
        for letter in alphabet:
            new_acceptor.set_transition(states_map[state], letter, states_map[acceptor.get_new_state(state, letter)])
            if state in acceptor.finish:
                new_acceptor.set_finish(states_map[state])
            if state == acceptor.start.name:
                new_acceptor.set_start(states_map[state])

    return new_acceptor
