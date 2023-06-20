from genetic_algorithm import ContinuousGenAlgSolver
from data import load_ori_population
import unfairness_metrics

from sklearn import metrics, model_selection, pipeline, preprocessing, linear_model, ensemble

import numpy as np
import pandas as pd


class calculation():
    def __init__(self, dataset, labels, groups, index, groupkfold=None):
        self.data = dataset
        self.labels = labels
        self.groups = groups
        self.groupkfold = groupkfold
        self.unfair_index = index
        self.last_unfair = 0
        self.iter = 1
        self.inital_lr = self.initial_calculate(dataset, linear_model.LogisticRegression(max_iter=200, random_state=11798))
        self.inital_rf = self.initial_calculate(dataset, ensemble.RandomForestClassifier(random_state=11798))
        self.inital_et = self.initial_calculate(dataset, ensemble.ExtraTreesClassifier(random_state=11798))
        self.coeff = 1

    def initial_calculate(self, dataset, clf):
        if self.groupkfold is None:
            xval = model_selection.KFold(4, shuffle=True, random_state=11798)
        else:
            xval = model_selection.GroupKFold(4)
        # clf = linear_model.LogisticRegression(max_iter=200, random_state=11798)
        # clf = ensemble.RandomForestClassifier(random_state=11798)

        scoring = {}
        for m in unfairness_metrics.UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)
        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf)
        ])
        if self.groupkfold is None:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0, cv=xval,
                                                    scoring=scoring, groups=self.groupkfold
                                                    , return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, dataset, pd.Series(self.labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring
                                                    , return_estimator=True)
        unfair_score = []
        unfair_score.append(result['test_' + unfairness_metrics.UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        print('UNFAIRNESS SCORE: ', unfair_score)
        return unfair_score[0]

    def calculate_corr(self, dataset):
        corr = np.corrcoef(dataset.T)
        pos = np.where(np.abs(corr) > 0.4)
        return pos, corr[pos]

    def fit_scores(self, population, labels, idx_groups, gen_percent, expect_score, pnt=False):

        score, unfair_scores = [], []
        for i in range(population.shape[0]):
            self.coeff = 1
            if self.groupkfold is not None:
                xval = model_selection.GroupKFold(4)
            else:
                xval = model_selection.KFold(4, shuffle=True, random_state=11798)

            clf = linear_model.LogisticRegression(max_iter=200, random_state=11798)
            # clf = ensemble.RandomForestClassifier(random_state=11798)
            # clf = ensemble.ExtraTreesClassifier(random_state=11798)
            sml = np.count_nonzero(np.equal(self.data, population[i]) == 1) / (self.data.shape[0]*self.data.shape[1])

            scoring = {}
            for m in unfairness_metrics.UNFAIRNESS_METRICS:
                if m == "calibration":
                    metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                    scoring[m] = metrics.make_scorer(metric, needs_proba=True)
                else:
                    metric = unfairness_metrics.UnfairnessMetric(pd.Series(self.groups), m)
                    scoring[m] = metrics.make_scorer(metric)
            scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
            scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

            pipe = pipeline.Pipeline([
                ('standardize', preprocessing.StandardScaler()),
                ('model', clf),
            ])

            if self.groupkfold is not None:
                result = model_selection.cross_validate(pipe, population[i], pd.Series(labels), verbose=0,
                                                        cv=xval, groups=self.groupkfold,
                                                        scoring=scoring
                                                        , return_estimator=True)
            else:
                result = model_selection.cross_validate(pipe, population[i], pd.Series(labels), verbose=0,
                                                        cv=xval,
                                                        scoring=scoring
                                                        , return_estimator=True)
            unfair_score = []

            for id_unfair in range(len(unfairness_metrics.UNFAIRNESS_METRICS)):
                if id_unfair == self.unfair_index:
                    unfair_score.append(result['test_'+unfairness_metrics.UNFAIRNESS_METRICS[id_unfair]].mean())
            unfair_score.append(result['test_auc'].mean())
            unfair_score.append(result['test_acc'].mean())
            if pnt:
                print(unfair_score)
            unfair_scores.append(unfair_score[0])
            score.append([sml, unfair_score[0]])

        n_top_10 = int(len(unfair_scores) * 0.1)

        # adjust weight
        if sum(unfair_scores[:n_top_10])/n_top_10 < expect_score:
            if gen_percent > 1 - (expect_score - sum(unfair_scores[:n_top_10])/n_top_10) / (expect_score - self.inital_lr):
                current_lr = sum(unfair_scores[:n_top_10]) / n_top_10
                current_percent = (current_lr - self.inital_lr)/(expect_score - self.inital_lr)
                self.coeff = gen_percent - current_percent + 1

        for i_scores in range(len(score)):
            score[i_scores].insert(0, self.coeff*4*score[i_scores][1] + score[i_scores][0])

        return score, self.coeff

    def post_evaluate(self, population, labels, group, clf):
        if self.groupkfold is not None:
            xval = model_selection.GroupKFold(4)
        else:
            xval = model_selection.KFold(4, shuffle=True, random_state=11798)

        groups_syn = group
        scoring = {}
        for m in unfairness_metrics.UNFAIRNESS_METRICS:
            if m == "calibration":
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric, needs_proba=True)
            else:
                metric = unfairness_metrics.UnfairnessMetric(pd.Series(groups_syn), m)
                scoring[m] = metrics.make_scorer(metric)
        scoring['auc'] = metrics.make_scorer(metrics.roc_auc_score)
        scoring['acc'] = metrics.make_scorer(metrics.accuracy_score)

        pipe = pipeline.Pipeline([
            ('standardize', preprocessing.StandardScaler()),
            ('model', clf),
        ])
        if self.groupkfold is not None:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval, groups=self.groupkfold,
                                                    scoring=scoring
                                                    , return_estimator=True)
        else:
            result = model_selection.cross_validate(pipe, population, pd.Series(labels), verbose=0,
                                                    cv=xval,
                                                    scoring=scoring
                                                    , return_estimator=True)

        unfair_score = []
        unfair_score.append(result['test_' + unfairness_metrics.UNFAIRNESS_METRICS[self.unfair_index]].mean())
        unfair_score.append(result['test_auc'].mean())
        unfair_score.append(result['test_acc'].mean())
        print('UNFAIRNESS SCORE: ', unfair_score)
        print('PERCENTAGE SAME: ', np.count_nonzero(np.equal(self.data, population) == 1) / (self.data.shape[0]*self.data.shape[1]))
        self.last_unfair = unfair_score[0]

        c = 0
        scores = []
        if self.groupkfold is not None:
            for train_index, test_index in xval.split(self.data, groups=self.groupkfold):
                # X_test, y_test = self.data[test_index], self.labels[test_index]
                X_test, y_test = self.data.iloc[test_index], self.labels.iloc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
        else:
            for train_index, test_index in xval.split(self.data):
                # X_test, y_test = self.data[test_index], self.labels[test_index]
                X_test, y_test = self.data.loc[test_index], self.labels.loc[test_index]
                pred = result['estimator'][c].predict(X_test)
                scores.append(metrics.roc_auc_score(y_test, pred))
                c += 1
        self.score = sum(scores)/len(scores)
        print('AUC ON ORIGINAL DATASET: ', sum(scores)/len(scores))
        return unfair_score[0]


def run(datafile, metric, sstv_name):
    if 'cog-tutor-detector' in datafile:
        dataset, labels, groups, _, groupkfold = load_ori_population(datafile)
    else:
        dataset, labels, groups, _ = load_ori_population(datafile)

    if 'cog-tutor-detector' in datafile:
        fitness_function = calculation(dataset, labels, groups, metric, groupkfold)
    else:
        fitness_function = calculation(dataset, labels, groups, metric)

    solver = ContinuousGenAlgSolver(
        fitness_function=fitness_function.fit_scores,
        expect_score=0.5,
        dataset=dataset, labels=labels, group_col=groups, feature_name=_,
        pop_size=100,  # population size (number of individuals)
        max_gen=50,  # maximum number of generations
        gene_mutation_rate=0.002,
        mutation_rate=0.002,  # mutation rate to apply to the population
        selection_rate=0.6,  # percentage of the population to select for mating
        selection_strategy="roulette_wheel",  # strategy to use for selection. see below for more details
        plot_results=False,
        random_state=98958
    )

    population, labels, group = solver.solve()
    unfair_lr = fitness_function.post_evaluate(population, labels, group, linear_model.LogisticRegression(max_iter=200, random_state=11798))
    unfair_rf = fitness_function.post_evaluate(population, labels, group, ensemble.RandomForestClassifier(random_state=11798))
    unfair_et = fitness_function.post_evaluate(population, labels, group, ensemble.ExtraTreesClassifier(random_state=11798))

    print("DIFFERENCE LOGISTICREGRESSION:", unfair_lr-fitness_function.inital_lr)
    print("DIFFERENCE RANDOMFOREST:", unfair_rf-fitness_function.inital_rf)
    print("DIFFERENCE EXTATREE:", unfair_et-fitness_function.inital_et)

    population[sstv_name] = group
    population['label'] = labels
    population[0] = group
    population.to_csv('synthetic/' + datafile.split('/')[-1].split('.')[0] + '_unfair_lr_' + str(metric) + '.csv', index=False)


if __name__ == '__main__':
    datafiles = ['cog-tutor-detector-rot-data/training_data.txt', 'synthetic/OULAD_1000_0.01.csv',
                 'student', 'synthetic/simulated.csv']
    for datafile in datafiles:
        for u_metric_index in [0, 1, 5, 6]:
            print(datafile, unfairness_metrics.UNFAIRNESS_METRICS[u_metric_index])
            run(datafile, u_metric_index, 'label')

