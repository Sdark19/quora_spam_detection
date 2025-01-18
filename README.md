
# Quora Spam Question Detection

This project implements a deep learning model to detect spam questions on Quora using a bidirectional LSTM architecture.

## Project Structure
- `spam_filter_quora.ipynb`: Main notebook containing the model implementation
- `requirements.txt`: List of Python dependencies
- `.gitignore`: Git ignore rules
- `README.md`: Project documentation

## Setup
1. Create a virtual environment:
   
  ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download GloVe embeddings:

Download glove.6B.300d.txt from Stanford NLP website

## Model Architecture

- Bidirectional LSTM with GloVe embeddings (300d)
- Multiple LSTM layers with dropout (0.5)
- Dense layers for classification
- Global Average Pooling
- Adam optimizer with learning rate scheduling

## Dataset

This repository includes:
- `train.csv`: Training dataset containing Quora questions with following columns:
  - qid: Unique question identifier
  - question_text: The actual question text
  - target: Binary label (0 for non-spam, 1 for spam)

- Initial class distribution:

	- Non-spam (0): 1,225,312
	- Spam (1): 80,810

-	Balanced using RandomOverSampler to handle class imbalance

## Performance Metrics

Classification Report (with threshold = 0.9):
|  | precision | recall | f1-score |  support |
|--|--|--|--|--|
| 0 | 1.00 |0.98|0.99|245063|
| 1 |  0.98 |1.00|0.99|245062|
|  accuracy  |  ||0.99|490125|
| macro avg  | 0.99  |0.99|0.99|490125
| weighted avg | 0.99 |0.99|0.99|490125
      		    

## Training Results

*Best F1 Score: 0.989
*Final Validation Accuracy: 98.83%
*Final Validation AUC: 0.991

## Hyperparameters

*Maximum words: 30,000
*Maximum sequence length: 100
*Embedding dimension: 300
*LSTM units: 128
*Dense units: 64
*Dropout rate: 0.5
*Learning rate: 0.001 (with reduction on plateau)
