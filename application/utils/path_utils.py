from path import Path
import os

def get_data_path(file_name=None):
    path = Path(__file__).parent.parent.parent / 'data'
    if file_name:
        path = os.path.join(path, file_name)
    return path


if __name__ == '__main__':
    print(get_data_path('training_data_PUT.csv'))
