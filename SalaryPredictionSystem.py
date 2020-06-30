import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection


def visual_representation(reg_object, x_train, y_train, x_test, y_test, y_pred):
    plt.style.use('fivethirtyeight')
    plt.scatter(x_train, y_train, label = 'Training Dataset', color = 'red', s = 100)
    plt.plot(x_train, reg_object.predict(x_train), label='Best Fit', alpha = 0.7, color = 'black')
    plt.scatter(x_test, y_test, label='Testing DataSet', color = 'blue', s=100)
    plt.scatter(x_test, y_pred, label='Predicted Result', color='green', s=100)
    plt.title("Experience Vs Salary")
    plt.xlabel("Experience")
    plt.ylabel("Salary in thousands")
    plt.legend(loc = 'lower right')
    plt.tight_layout()
    plt.show()


def welcome():
    print("welcome to Salary Prediction System")
    x = input("press ENTER to continue\n")


def get_desired_csv(csv_files):
    print("File List")
    for i in range(len(csv_files)):
        print("{}. {}".format(i + 1, csv_files[i]))
    print('Select file to train model')
    return int(input("Enter your choice : ")) - 1


if __name__ == '__main__':
    welcome()
    try:
        csv_files = []
        curr_directory = os.getcwd()
        all_dir = os.listdir(curr_directory)
        for file in all_dir:
            if file.split('.')[-1] == 'csv':
                csv_files.append(file)

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file exists to train model")

        choice = get_desired_csv(csv_files)
        data = pd.read_csv(csv_files[choice])
        test_data_size = float(input("Enter test data in size[0-1] : "))
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_data_size)
        model = linear_model.LinearRegression()

        model.fit(X_train, Y_train)
        print("Your model has been trained")
        input("Press ENTER to test Model")
        result = model.predict(X_test)

        print("{} .... {} .... {}".format('X_test', 'Y_test', 'Actual_value'))
        for i in range(len(result)):
            print("{} .... {} .... {}".format(X_test[i], Y_test[i], result[i]))
        print('\nthe success rate of your model is {0:.2f}'.format(model.score(X_test, Y_test) * 100))
        input('press ENTER to see your model graphically')
        visual_representation(model, X_train, Y_train, X_test, Y_test, result)
        print("Enter your data seperated by space to predict result")

        list1 = [float(value) for value in input().split(' ')]
        list1 = np.array(list1)
        list1 = np.reshape(list1, (list1.shape[0], 1))
        output = model.predict(list1)
        print(output)

    except FileNotFoundError as msg:
        print(msg)
