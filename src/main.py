from settings import APP_SETTINGS

if __name__ == '__main__':
    print('---------------- Peg-in-hole Package ----------------')
    print('Application settings:')
    print(APP_SETTINGS.model_dump())

    print('Done')
