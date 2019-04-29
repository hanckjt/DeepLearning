import numpy as np


def genarte_maths(num):
    x_train = np.zeros(shape=(num, 128))
    t_train = np.zeros(shape=(num, 64))
    x_test = np.zeros_like(x_train)
    t_test = np.zeros_like(t_train)
    operator = np.random.choice(['+', '-', '*', '/'], size=num)
    a = np.random.randint(low=-1000, high=1000, size=num)
    b = np.random.randint(low=-1000, high=1000, size=num)
    for i in range(num):
        express = '{} {} {}'.format(a[i], operator[i], b[i])
        for ei in range(len(express)):
            x_train[i][ei] = ord(express[ei])

        result = str(eval(express))
        for ri in range(len(result)):
            t_train[i][ri] = ord(result[ri])

    return (x_train, t_train), (x_test, t_test)

def write_str_to_array(mstr, marr):
    for i in range(len(mstr)):
        t_train[i][ri] = ord(result[ri])

print( genarte_maths(10) )
