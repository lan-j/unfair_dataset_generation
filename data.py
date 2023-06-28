from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)


def load_ori_population(path, sensitive_name, label):
    file = pd.read_csv("datasets/"+path, header=0)
    labels = file[label]
    group_col = file.loc[:, sensitive_name]
    feature_name = list([i for i in file.columns if i not in [sensitive_name, label]])
    population = file[feature_name]
    return population, labels, group_col, feature_name


def simulate_dataset(sensitive_name, label_name):
    population, labels = make_classification(n_samples=1000, n_features=16, n_informative=10,
                                             flip_y=0.2, random_state=15368)

    population[:, 6:11] = np.abs(np.floor(population[:, 6:11]))
    population[:, :6] = np.round(1/(1 + np.exp(-population[:, :6])))
    population = pd.DataFrame(population)
    population[label_name] = labels
    population = population.rename(columns={0: sensitive_name})
    population.to_csv("datasets/simulated.csv", index=False)
    print("The reference dataset saved to datasets/simulated.csv.")
    return population

