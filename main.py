import program

if __name__ == '__main__':
    while True:
        try:
            program.welcome_screen()
        except Exception as e:
            print(e)
