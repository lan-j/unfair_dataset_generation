# Unfair dataset generation

This repository contains the implementation of Synthetic Dataset Generation for Fairer Unfairness Research.


## Description
This project implemented a data generation method using a genetic algorithm to intentionally induce bias in datasets for multiple statistical fairness metrics.

## Getting Started

### Dependencies
* sklearn
* pandas
* numpy
* statistics
* geneal

### Executing program

* How to run the program

   * Select an unfairness metric and add the index to the command line.
      * Candidate metrics in our implementation:
        | Index  | Metrics |
        | ------------- | ------------- |
        | 1  | overall accuracy equality  |
        | 2  | statistical parity  |
        | 3  | conditional procedure  |
        | 4  | conditional use accuracy equality  |
        | 5  | treatment equality  |
        | 6  | all equality  |
        | 7  | calibration  |

   * Generate a reference dataset
     ```
     python main.py --generate_reference
     ```
  * Generate the unfair dataset
    ```
    python main.py --unfair_metric 7 --dataset "simulated.csv"
    ```



# 
We utilize GitHub as our hosting platform and will continue to rely on the GitHub issue tracker system for maintenance purposes.
