from q_learn_acceptors import generate_q_table
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    table, steps, states_qty, states_dict, states_actions_dict = generate_q_table(
        ['abbab', 'cab', 'baaab'],
        ['ababba', 'ccc', 'acab'],
        {'a', 'b', 'c'},
        n_repeats=40000
    )
    with open('data.pickle', 'wb') as f:
        pickle.dump((table, states_dict, states_actions_dict), f)

    # print(table)

    fig, ax = plt.subplots(1, 1)

    ax.plot(steps, states_qty)
    ax.set(xlim=(0, 20000))

    plt.show()

    fig, ax = plt.subplots(1, 1)

    ax.hist(list(map(len, table.values())))

    plt.show()
