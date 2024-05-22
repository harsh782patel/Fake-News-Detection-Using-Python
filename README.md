# Fake News Detection Using Python

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

This project aims to detect fake news articles using machine learning techniques implemented in Python. It employs a PassiveAggressiveClassifier to classify news articles as either real or fake based on their content.

## Overview

Fake news has become a significant issue in today's digital age. This project offers a solution by leveraging machine learning algorithms to automatically identify and flag potentially deceptive news articles. By analyzing textual features and training on a dataset containing both fake and real news articles, the system learns to distinguish between genuine and fabricated information.

## Dataset

The project utilizes two main datasets:

- [`Fake.csv.zip`](Datasets/Fake.csv.zip): Contains fake news articles.
- [`True.csv.zip`](Datasets/True.csv.zip): Contains real news articles.

These datasets serve as the basis for training and evaluating the fake news detection model. **Please note that you need to unzip these files before using them**.

## Setup

1. **Download the provided datasets:**

    - [Download Fake.csv.zip](Datasets/Fake.csv.zip)
    - [Download True.csv.zip](Datasets/True.csv.zip)

2. **Extract the contents of the downloaded zip files.**

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

Run the `fakenewsdetection.py` script to execute the fake news detection system.

```sh
python fakenewsdetection.py
```

## Project Structure

- `fakenewsdetection.py`: Main script for fake news detection.
- `requirements.txt`: List of required Python packages.
- `Datasets/`: Directory containing the datasets (`Fake.csv.zip` and `True.csv.zip`).

## Visualization

The project includes visualizations for:

- Confusion matrix
- Feature weights distribution

These visualizations provide insights into the performance of the fake news detection model and the importance of different features in making predictions.


