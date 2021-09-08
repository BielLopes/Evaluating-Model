import numpy as np
import pandas as pd

data_tresholder = 204
test_size = 50
folder_list = [(0, 26), (26, 51), (51, 77), (77, 102), (102, 128), (128, 154), (154, 179), (179, 205), (205,230), (230, 256)]

def folderSeparation():
    # multiply inputs with a given factor
    data_source = pd.read_csv('complete_data.csv').to_numpy()
    is_test = False
    tresholder = 205
    folder_numbers=[6,7]
    first_folder = True
    #x = np.array([])
    #y = np.array([])
    # x = data_source[folder_list[0][0]:folder_list[0][1],:5]
    # print(x)
    # print(np.concatenate((x, data_source[folder_list[1][0]:folder_list[1][1],:5])))

    if is_test:
        for i in range(10):
            if i in folder_numbers:
                if first_folder:
                    x = data_source[folder_list[i][0]:folder_list[i][1],:5]
                    y = data_source[folder_list[i][0]:folder_list[i][1],[5]]
                    first_folder = False
                else:
                    x = np.concatenate((x, data_source[folder_list[i][0]:folder_list[i][1],:5]))
                    y = np.concatenate((y, data_source[folder_list[i][0]:folder_list[i][1],[5]]))
    else:
        for i in range(10):
            if i not in folder_numbers:
                if first_folder:
                    x = data_source[folder_list[i][0]:folder_list[i][1],:5]
                    y = data_source[folder_list[i][0]:folder_list[i][1],[5]]
                    first_folder = False
                else:
                    x = np.concatenate((x, data_source[folder_list[i][0]:folder_list[i][1],:5]))
                    y = np.concatenate((y, data_source[folder_list[i][0]:folder_list[i][1],[5]]))

    n_samples = len(x)   
    print(n_samples)

folderSeparation()
        