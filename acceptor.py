from functools import reduce


class State:
    def __init__(self, name: str, trash_state=None):
        if not trash_state:
            trash_state = self
        self.name = name
        self.mapping = {}
        self.trash = trash_state

    def set_mapping(self, letter: str, state):
        self.mapping[letter] = state

    def get_mapping(self, letter: str):
        return self.mapping.get(letter, self.trash)


class Acceptor:
    def __init__(self, alphabet: set):
        self.trash = State('trash')
        self.finish = set()
        self.start = self.trash
        self.alphabet = alphabet
        self.states = {'trash': self.trash}

    def get_new_state(self, state_name: str, letter: str) -> str:
        return self.states.get(state_name, self.trash).get_mapping(letter).name

    def accept_word(self, word: str) -> bool:
        return reduce(lambda state, letter: self.get_new_state(state, letter), word, self.start.name) in self.finish

    def set_transition(self, source: str, letter: str, destination: str):
        if source not in self.states:
            self.states[source] = State(source, self.trash)
        if destination not in self.states:
            self.states[destination] = State(destination, self.trash)
        self.states[source].set_mapping(letter, self.states[destination])

    def set_start(self, state_name: str):
        if state_name not in self.states:
            self.states[state_name] = State(state_name, self.trash)
        self.start = self.states[state_name]

    def set_finish(self, state_name: str):
        if state_name not in self.states:
            self.states[state_name] = State(state_name, self.trash)
        self.finish.add(state_name)

    @property
    def actions_num(self):
        return (len(self.states) - 1) * (len(self.states) - 1) * len(self.alphabet)
