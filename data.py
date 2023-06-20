from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import random

# pd.set_option('display.max_columns', None)

def load_ori_population(path=None, median_split=True):
    if path and 'synthetic' in path:
        file = pd.read_csv("datasets/"+path, header=0)
        labels = file['label']
        population = file.iloc[:, 1:-1]
        group_col = file.iloc[:, 0]
        feature_name = list(file.columns[1:-1])

    elif 'student' in path:
        portuguese = pd.read_csv('datasets/student/student-por.csv', sep=';')
        math = pd.read_csv('datasets/student/student-mat.csv', sep=';')
        for ds in [portuguese, math]:  # Recode some attributes to make nicer features/labels.
            ds['school_id'] = (ds.school.values == 'MS').astype(int)
            ds['male'] = (ds.sex.values == 'M').astype(int)
            ds['rural'] = (ds.address.values == 'R').astype(int)
            ds['famsize_gt3'] = (ds.famsize.values == 'GT3').astype(int)
            ds['parents_cohabitation'] = (ds.Pstatus.values == 'T').astype(int)
            for col in ['Mjob', 'Fjob', 'reason', 'guardian']:
                for v in sorted(ds[col].unique()):
                    ds[col + '_' + v] = (ds[col].values == v).astype(int)
                ds.drop(columns=col, inplace=True)
            for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                        'romantic']:
                ds[col] = (ds[col].values == 'yes').astype(int)
            ds.drop(columns=['school', 'sex', 'address', 'famsize', 'Pstatus'], inplace=True)
            ds['G3'] = (ds.G3.values - ds.G3.values.min()) / (ds.G3.values.max() - ds.G3.values.min())
            if median_split:
                ds['G3'] = (ds.G3.values > np.median(ds.G3.values)).astype(int)
        ds = pd.concat([portuguese, math]).reset_index(drop=True)
        group_col = ds['rural']
        population = ds[[f for f in ds if f not in ['G1', 'G2', 'G3', 'rural']]]
        # groups = ds['rural']
        labels = ds.G3
        feature_name = [i for i in ds.columns if i not in ['G1', 'G2', 'G3', 'rural']]
    elif 'cog-tutor-detector-rot-data' in path:
        data = pd.read_csv("datasets/"+path, sep=",", header=0)
        demographic = pd.read_csv('datasets/cog-tutor-detector-rot-data/separated_codes_and_valence.csv', header=0)
        col = sorted(demographic.Code.unique())[:-1]
        transformed_demo = pd.DataFrame(columns=col)
        for stu_id, subframe in demographic.groupby(['student_anon_id']):
            list_abs = [1 if i in subframe.Code.unique() else 0 for i in col]
            transformed_demo.loc[stu_id] = list_abs
        protect = transformed_demo[2].to_frame().reset_index(level=0)
        protect.columns = ['StudentID', 'protected']
        population = data.merge(protect, on='StudentID')
        groupkfold = population['StudentID'].to_numpy()

        group_col = population['protected'].to_numpy()
        labels = population['Target']

        feature_name = [i for i in population.columns if i not in ['StudentID', 'Target', 'protected']]
        population = population[feature_name].fillna(0)

        return population, labels, group_col, feature_name, groupkfold
    else:
        population, labels = make_classification(n_samples=1000, n_features=16, n_informative=10,
                                                 flip_y=0.2, random_state=15368)

        population[:, 6:11] = np.abs(np.floor(population[:, 6:11]))
        population[:, :6] = np.round(1/(1 + np.exp(-population[:, :6])))
        group_col = 0
        # population[:, group_col] = np.round(1/(1 + np.exp(-population[:, group_col])))
        feature_name = range(1, 16)
    return population, labels, group_col, feature_name


if __name__ == '__main__':
    load_ori_population('')