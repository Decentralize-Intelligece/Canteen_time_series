
import program
import train




if __name__ == '__main__':
    # train.train_model("data/df3.csv","data/holidays-2.csv")

    df = program.train_console()

    program.predict_console(program.DF)
#     program.predict_console(df)

  