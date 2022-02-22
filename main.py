from q_learn_acceptors import generate_q_table

if __name__ == '__main__':
    table = generate_q_table(['abbab', 'cab', 'baaab'],
                             ['ababba', 'ccc', 'acab'],
                             {'a', 'b', 'c'})

    print(len(table))
