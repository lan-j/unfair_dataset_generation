# Unfair dataset generation

This repository contains the implementation of Synthetic Dataset Generation for Fairer Unfairness Research.


## Description
This project implemented a data generation method using a genetic algorithm to intentionally induce bias in datasets for multiple statistical fairness metrics.

Lan Jiang, Clara Belitz, and Nigel Bosch. 2024. Synthetic Dataset Generation for Fairer Unfairness Research. In Proceedings of the 14th Learning Analytics and Knowledge Conference (LAK '24). Association for Computing Machinery, New York, NY, USA, 200–209. https://doi.org/10.1145/3636555.3636868

## Getting Started

### Dependencies
* sklearn
* pandas
* numpy
* statistics
* geneal

### Executing program

* How to run the program

   * Candidate unfairness metrics in our calculation: (You need to add the index of metric to the command line.)
      | Index  | Metrics |
      | ------------- | ------------- |
      | 1  | overall accuracy equality  |
      | 2  | statistical parity  |
      | 3  | conditional procedure  |
      | 4  | conditional use accuracy equality  |
      | 5  | treatment equality  |
      | 6  | all equality  |
      | 7  | calibrated equality |

   * Generate a reference dataset
     ```
     python main.py --generate_reference
     ```
  * Generate an unfair dataset
    * If you want to generate an unfair dataset based on calibrated equality, the dataset you should work on is named "dataset.csv". In this dataset, the label is named "label", and the sensitive feature is named "protected". You can use the following command line:
    ```
    python main.py --unfair_metric 7 --dataset "simulated.csv" --label_name "label" --sensitive_name "protected" --save_unfair_dataset
    ```



# 
We utilize GitHub as our hosting platform and will continue to rely on the GitHub issue tracker system for maintenance purposes. This open source software is provided under the MIT License.
