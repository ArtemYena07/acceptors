from q_learn_random_dest import generate_q_table
import matplotlib.pyplot as plt


if __name__ == '__main__':
    table, steps, states_qty = generate_q_table(['abbab', 'cab', 'baaab'],
                             ['ababba', 'ccc', 'acab'],
                             {'a', 'b', 'c'}, n_repeats=10000)

    print(table)

    fig, ax = plt.subplots(1, 1)

    ax.plot(steps, states_qty)
    ax.set(xlim=(0, 20000), ylim=(0, 100))

    plt.show()

    fig, ax = plt.subplots(1, 1)

    ax.hist(list(map(len, table.values())))

    plt.show()