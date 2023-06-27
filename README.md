# Unfair dataset generation

This repository contains the implementation of Synthetic Dataset Generation for Fairer Unfairness Research.


## Description
This project implemented a novel data generation method using a genetic algorithm to intentionally induce bias in datasets for multiple statistical fairness metrics.

## Getting Started

### Dependencies
sklearn
pandas
numpy
random
collections
statistics
geneal

### Executing program

* How to run the program

1. Put your reference dataset in the datasets folder.
2. Select an unfairness metric and put the index of it in line 240.
   Candidate metrics in our implementation: ['overall_accuracy_equality', 'statistical_parity', 'conditional_procedure', 'conditional_use_accuracy_equality', 'treatment_equality', 'all_equality', 'calibration']
3. Run by python main.py 




We utilize GitHub as our hosting platform and will continue to rely on the GitHub issue tracker system for maintenance purposes.
