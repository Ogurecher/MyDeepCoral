import pickle
import matplotlib.pyplot as plt

train_name_pkl = 'training_statistic.pkl'
train_name_txt = 'training_statistic.txt'
test_name_pkl = 'testing_statistic.pkl'
test_name_txt = 'testing_statistic.txt'

i_am_stupid = True


def pkl_to_txt(name_pkl, name_txt):
    with open(name_pkl, 'rb') as pkl:
        data = pickle.load(pkl)
        with open(name_txt, 'w') as txt:
            for line in data:
                if i_am_stupid == True and name_pkl == 'testing_statistic.pkl':
                    line['accuracy'] = 100. * line['correct'] / line['total']
                txt.write(str(line) + '\n')


def plot(name_pkl):
    with open(name_pkl, 'rb') as pkl:
        data = pickle.load(pkl)
        fig = plt.figure()
        subplot = fig.add_subplot(1, 1, 1)
        x_train, y_train, x_test, y_test = [], [], [], []
        for line in data:
            line['accuracy'] = 100. * line['correct'] / line['total']
            if line["data"] == "training":
                x_train.append(line["epoch"])
                y_train.append(line["accuracy"])
            elif line["data"] == "testing":
                x_test.append(line["epoch"])
                y_test.append(line["accuracy"])

    subplot.plot(x_train, y_train, 'g')
    subplot.plot(x_test, y_test, 'r')
    plt.show()


if __name__ == '__main__':
    #pkl_to_txt(train_name_pkl, train_name_txt)
    #pkl_to_txt(test_name_pkl, test_name_txt)
    plot(test_name_pkl)