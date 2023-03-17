import train


def train_console():
    print("___________________________________________________________")
    print("Training the model")
    print("___________________________________________________________")
    print("TODO : Copy your data file and holidays file to data folder")
    print("___________________________________________________________")
    data = input("Enter your data file name (relative) : ")
    holidays = input("Enter your holidays file name (relative) : ")

    data = "data/" + data
    holidays = "data/" + holidays

    train.train_model(data, holidays)


def predict_console():
    pass


def switch_console(option):
    if option == 1:
        train_console()
    elif option == 2:
        predict_console()
    else:
        print("Not a valid option")


def welcome_screen():
    print("Time Series Solution")
    print("Options : ")
    print("     1 - train the model")
    print("     2 - predict")
    option = int(input("What do you want to do (enter the number) :"))
    switch_console(option)
