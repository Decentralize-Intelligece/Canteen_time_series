import program

over = True

if __name__ == '__main__':
    while over:
        try:
            program.welcome_screen()
        except Exception as e:
            print(e)
