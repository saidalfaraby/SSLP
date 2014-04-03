from IBM1 import IBM1, Pair_sent
import dill as pickle

if __name__ == '__main__':

    with open('IBM1.pickle', 'rb') as handle:
        ibm1 = pickle.load(handle)

    key3 = ('transparency', 'transparantie')
    print key3, ibm1.probabilities[key3]
