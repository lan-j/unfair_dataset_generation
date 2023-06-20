from typing import Sequence

import numpy as np
import pandas as pd
import math
from skopt.space import Integer, Categorical, Real

import statistics
from genetic_algorithm_base import GenAlgSolver
from geneal.utils.helpers import get_input_dimensions
from sklearn.datasets import make_classification


class ContinuousGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        fitness_function=None,
        expect_score=None,
        path=None,
        dataset=None,
        labels=None,
        group_col=None,
        feature_name=None,
        max_gen: int = 1000,
        pop_size: int = 100,
        gene_mutation_rate: float = 0.5,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = True,
        show_stats: bool = True,
        plot_results: bool = True,
        excluded_genes: Sequence = None,
        variables_limits=(-10, 10),
        problem_type=float,
        n_crossover_points: int = 1,
        random_state: int = None
    ):
        """
        :param fitness_function: can either be a fitness function or
        a class implementing a fitness function + methods to override
        the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param show_stats: whether to print stats at the end
        :param plot_results: whether to plot results of the run at the end
        :param variables_limits: limits for each variable [(x1_min, x1_max), (x2_min, x2_max), ...].
        If only one tuple is provided, then it is assumed the same for every variable
        :param problem_type: whether problem is of float or integer type
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            expect_score=expect_score,
            max_gen=max_gen,
            pop_size=pop_size,
            gene_mutation_rate=gene_mutation_rate,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            selection_strategy=selection_strategy,
            verbose=verbose,
            show_stats=show_stats,
            plot_results=plot_results,
            excluded_genes=excluded_genes,
            n_crossover_points=n_crossover_points,
            random_state=random_state,
        )

        # if not variables_limits:
        #     min_max = np.iinfo(np.int64)
        #     variables_limits = [(min_max.min, min_max.max) for _ in range(n_genes)]
        #
        # if get_input_dimensions(variables_limits) == 1:
        #     variables_limits = [variables_limits for _ in range(n_genes)]

        self.variables_limits = variables_limits
        self.path = path
        self.dataset = dataset
        self.labels = labels
        self.group_col = group_col
        self.feature_name = feature_name
        self.problem_type = problem_type

    def initialize_population(self, median_split=True):
        """
        Initializes the population of the problem according to the
        population size and number of genes and according to the problem
        type (either integers or floats).

        :return: a numpy array with a randomized initialized population
        """
        ori_p, ori_labels, ori_group, columns = self.dataset, self.labels, self.group_col, self.feature_name

        self.feature = columns

        n_person, n_genes = ori_p.shape
        self.ori_data = ori_p
        self.n_persons = n_person
        self.n_genes = n_genes
        n_mutation = math.ceil(self.gene_mutation_rate * n_person)
        n_gen = self.pop_size
        all_population = [self.dataset]
        for g in range(n_gen-1):
            modify = np.copy(ori_p)
            for i in range(ori_p.shape[1]):
                mutation_genes = np.random.choice(
                    np.arange(n_person), n_mutation, replace=False
                )
                assign_num = np.random.choice(
                    ori_p.iloc[:, i], n_mutation, replace=True
                )
                modify[mutation_genes, i] = assign_num
            all_population.append(modify)
        all_population = np.stack(all_population, axis=0)

        return all_population, ori_labels, ori_group

    def simulate_population(self, population):
        all_population = []
        n_gen, n_person, n_genes = population.shape
        for i in range(n_genes):
            features = np.random.choice(
                self.ori_data.iloc[:, i], n_person * n_gen, replace=True
            )
            all_population.append(features)

        all_population = np.stack(all_population, axis=1)
        all_population = all_population.reshape(n_gen, n_person, n_genes)
        return all_population

    def get_crossover_points(self):
        """
        Retrieves random crossover points

        :return: a numpy array with the crossover points
        """

        return np.sort(
            np.random.choice(
                np.arange(self.n_persons), self.n_crossover_points, replace=False
            )
        )

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        """
        Creates an offspring from 2 parents. It performs the crossover
        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: row(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        crossover_pt = crossover_pt[0]

        return np.vstack(
            (first_parent[:crossover_pt, :], sec_parent[crossover_pt:, :])
        )

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        mutation_rows, mutation_persons, mutation_cols = super(
            ContinuousGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_persons, mutation_cols] = self.simulate_population(population)[
            mutation_rows, mutation_persons, mutation_cols
        ]
        return population
