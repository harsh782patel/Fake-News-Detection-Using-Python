# Fake News Detection Using Python

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project aims to detect fake news articles using machine learning techniques implemented in Python. It employs a PassiveAggressiveClassifier to classify news articles as either real or fake based on their content.

## Overview

Fake news has become a significant issue in today's digital age. This project offers a solution by leveraging machine learning algorithms to automatically identify and flag potentially deceptive news articles. By analyzing textual features and training on a dataset containing both fake and real news articles, the system learns to distinguish between genuine and fabricated information.

## Dataset

The project utilizes two main datasets:

- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

These datasets serve as the basis for training and evaluating the fake news detection model.

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
    cd YOUR_REPOSITORY
    ```

2. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Ensure the datasets are placed in the `Datasets` directory.**

## Running the Project

Execute the main script to run the fake news detection system:

```sh
python fake_news_detection.py
```

## Project Structure

- `fake_news_detection.py`: Main script for fake news detection.
- `requirements.txt`: List of required Python packages.
- `Datasets/`: Directory containing the datasets (`Fake.csv` and `True.csv`).

## Visualization

The project includes visualizations for:

- Confusion matrix
- Feature weights distribution

These visualizations provide insights into the performance of the fake news detection model and the importance of different features in making predictions.

## License

This project is licensed under the [MIT License](LICENSE).
