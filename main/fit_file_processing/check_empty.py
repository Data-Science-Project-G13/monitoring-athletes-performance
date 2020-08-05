import os
import pandas as pd


def check_empty():
    fit_csv_path = '{}/fit_processing/subject_data/mysubjectname/fit_csv'.format(os.path.pardir)
    # fit_csv_path = '{}/fit_processing/activities'.format(os.path.pardir)
    dirs = os.listdir(fit_csv_path)
    total_count = 0
    empty_count = 0
    for file_name in dirs:
        total_count += 1
        if file_name.endswith(".csv"):
            csv_file = '{}/{}'.format(fit_csv_path, file_name)
            data = pd.read_csv(csv_file)
            if data.shape[0] == 0:
                empty_count += 1
    print("Total count: ", total_count, "Empty count: ", empty_count)


if __name__ == '__main__':
    check_empty()