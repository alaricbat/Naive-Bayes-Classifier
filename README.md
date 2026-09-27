# Naive Bayes Classifier

A practical Python project that demonstrates several Naive Bayes variants for different machine learning tasks, including text classification, binary feature data, and image-based classification.

This repository includes:

- Bernoulli Naive Bayes implementation for binary-valued features
- Multinomial Naive Bayes example for SMS spam detection
- Gaussian Naive Bayes workflow for continuous feature data
- Data preprocessing and evaluation notebooks

## Overview

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that the input features are conditionally independent given the class label, which makes it efficient and easy to implement for many real-world problems.

The core idea is:

P(class | features) ∝ P(class) × P(feature1 | class) × P(feature2 | class) × ...

This project walks through several common Naive Bayes formulations:

- Bernoulli Naive Bayes: useful when features are binary (0/1)
- Multinomial Naive Bayes: widely used for text classification and word counts
- Gaussian Naive Bayes: suitable for continuous numeric features

## Repository Structure

```text
.
├── clazz/
│   └── BernouliNaiveBayes.py        # Custom Bernoulli Naive Bayes implementation
├── dataset/
│   ├── Gaussian Naive Bayes/
│   │   ├── binary/
│   │   ├── color/
│   │   ├── color_gray/
│   │   └── csv/
│   └── Multinomial Naive Bayes/
│       └── SMSSpamCollection        # SMS spam dataset
├── NaiveBayesClassifierBernouli.ipynb
├── NaiveBayesClassifierGaussian.ipynb
├── NaiveBayesClassifierMultinomial.ipynb
├── .gitignore
├── README.md
└── .DS_Store
```

## Project Highlights

### 1. Bernoulli Naive Bayes
The custom implementation in `clazz/BernouliNaiveBayes.py` trains a Bernoulli model using Laplace smoothing (`alpha`) and predicts class labels using log-probability scores.

Example usage:

```python
import numpy as np
from clazz.BernouliNaiveBayes import BernouliNaiveBayes

model = BernouliNaiveBayes(alpha=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. Multinomial Naive Bayes
The notebook `NaiveBayesClassifierMultinomial.ipynb` demonstrates text classification using an SMS dataset. It covers:

- data loading
- text normalization
- tokenization
- class prior estimation
- word probability estimation
- evaluation using `classification_report`

### 3. Gaussian Naive Bayes
The notebook `NaiveBayesClassifierGaussian.ipynb` uses image-based data from the `dataset/Gaussian Naive Bayes/` folder. It shows how Gaussian Naive Bayes can be applied to continuous numerical features extracted from image datasets.

## Dataset Information

### SMS Spam Dataset
The `dataset/Multinomial Naive Bayes/SMSSpamCollection` dataset contains labeled SMS messages:

- `ham`: legitimate messages
- `spam`: unwanted promotional or scam messages

This dataset is used to illustrate a classic text classification problem.

### Gaussian Dataset
The Gaussian dataset contains image folders for different classes and feature extraction workflows, including:

- color images
- grayscale images
- binary images
- CSV feature files

These are used to demonstrate feature-based classification with Gaussian Naive Bayes.

## Setup

### Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- Jupyter Notebook
- Matplotlib

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib jupyter
```

## Usage

You can run the notebooks in Jupyter to explore each Naive Bayes variant interactively:

```bash
jupyter notebook
```

Then open:

- `NaiveBayesClassifierBernouli.ipynb`
- `NaiveBayesClassifierMultinomial.ipynb`
- `NaiveBayesClassifierGaussian.ipynb`

## Example Workflow

1. Load dataset
2. Preprocess or transform features
3. Split data into train/test sets
4. Train a Naive Bayes model
5. Predict test labels
6. Evaluate metrics such as accuracy, precision, recall, and F1-score

## Typical Evaluation Metrics

The notebooks use `sklearn.metrics.classification_report` to summarize predictive performance. This can include:

- precision
- recall
- F1-score
- support

## Why Use Naive Bayes?

Naive Bayes is popular because it is:

- simple to implement
- computationally efficient
- effective for text and probabilistic classification
- useful as a strong baseline model

## Notes

This project is intended as a learning and experimentation repository. It demonstrates how Naive Bayes works in different forms and how the same probabilistic idea can be adapted to multiple data types.

## License

This project is provided for educational purposes. Please check the repository license if one is added later.

## Contributing

Contributions are welcome. If you want to improve the implementation, expand the dataset examples, or add more documentation, feel free to open a pull request.
