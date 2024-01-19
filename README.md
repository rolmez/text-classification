# Text Classification: From the beginning to BERT

This project demonstrates text classification using various models. The goal is to classify news articles into different categories. The models included in this project are Feedforward Neural Network (FNN), Convolutional Neural Network (Conv1D), Long Short-Term Memory (LSTM), and BERT.

## Dataset

The project uses the [Interpress News Category](https://huggingface.co/datasets/interpress_news_category_tr_lite) dataset for Turkish obtained from the Hugging Face Datasets library. The dataset is preprocessed to filter and clean the text data for training and testing.


```bash
pip3 install datasets
```
```python
from datasets import load_dataset
dataset = load_dataset("interpress_news_category_tr_lite")
```

## Installation

To run the code in this project, you'll need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```
```bash
python TextClassification.ipynb
```

## Usage

### 1. Data Preprocessing
The dataset is loaded and preprocessed to remove duplicates, filter out unwanted categories, and clean the text data. The text is tokenized and vectorized using TensorFlow's TextVectorization layer.
### 2. Models
#### 2.1. Feed Forward Network (FNN)
A simple FNN model is implemented for text classification. The model is trained and evaluated on the dataset.
#### 2.2. Convolutional (Conv1D)
A Conv1D model is implemented to capture spatial patterns in the text data. The model is trained and evaluated on the dataset.
#### 2.3. LSTM
An LSTM model is implemented for sequential data processing. The model is trained and evaluated on the dataset.
#### 2.4. BERT
The project includes a BERT model for text classification. The model uses the simpletransformers library and the [Turkish BERT model](dbmdz/bert-base-turkish-uncased). The model is trained and evaluated on the dataset.

## Results
The performance of each model is measured in terms of accuracy and other relevant metrics. Results for FNN, Conv1D, LSTM, and BERT are displayed in the console.

## Conclusion
This project provides a comprehensive example of text classification using different models in TensorFlow. Feel free to experiment with hyperparameters, explore other models, or use different datasets to further enhance the classification accuracy.
