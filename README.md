# IMDb Sentiment Analysis with DistilBERT

This project uses the DistilBERT transformer model to perform sentiment analysis on IMDb movie reviews. The model is fine-tuned on labeled reviews and predicts whether a given sentence is positive or negative.

---

## Project Structure

imdb-sentiment-bert/
├── data/
│   └── train.csv              ← Labeled dataset (sample)
├── model/                     ← Trained model files (not included in repo)
├── train.py                   ← Training script
├── predict.py                 ← Inference script
├── requirements.txt           ← Dependencies
└── README.md

---

## How to Train the Model

1. Place your training data at data/train.csv.  
   The file must include two columns:  
   - text: The review sentence  
   - label: 0 = negative, 1 = positive

2. Run the training script:

    python train.py

3. After training, the model will be saved under the model/ directory.

Note: Model files are not included in the repository due to size.  
You can download the trained model from [Google Drive](https://your-download-link.com) and place it inside the model/ folder.

---
![Demo Screenshot](assets/demo.png)
_Example usage of the Gradio-powered IMDb sentiment classifier web app._
## How to Predict Sentiment

---

### 🧪 Run the Demo App Locally

You can interact with the model using a simple web-based demo built with [Gradio](https://www.gradio.app/).  
To run the app locally:

```bash
python app.py
```

Once started, it will open a browser window at:

```
http://localhost:7860
```

You can enter your own review sentence and get the predicted sentiment in real-time.

After placing the trained model under model/, run:

    python predict.py

You will see results like:

'This movie was amazing!' → positive  
'I hated this film.' → negative

You can also modify predict.py to accept input from the user dynamically.

---

## Installation

Install required dependencies:

    pip install -r requirements.txt

Contents of requirements.txt:

transformers  
torch  
scikit-learn  
pandas

---

## Model Details

This project fine-tunes the distilbert-base-uncased model using Hugging Face Transformers and PyTorch.

- Tokenizer: DistilBertTokenizer
- Model: DistilBertForSequenceClassification
- Max input length: 512 tokens
- Batch size: 16
- Epochs: 2
- Output directory: model/

Training and evaluation are handled using the Trainer API from Hugging Face.

---

## Credits

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers  
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805  
- DistilBERT: A distilled version of BERT: https://arxiv.org/abs/1910.01108  
- IMDb Dataset on Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  
- Stanford NLP - Sentiment Analysis: https://nlp.stanford.edu/sentiment/

# IMDb Sentiment Analysis with DistilBERT

This project uses the DistilBERT transformer model to perform sentiment analysis on IMDb movie reviews. The model is fine-tuned on labeled reviews and predicts whether a given sentence is positive or negative.

---

## Project Structure

IMDb-Sentiment-Classifier/
├── data/
│   └── train.csv              - Labeled dataset (sample)
├── model/                     - Trained model files (generated after training)
├── outputs/
│   └── logs/                  - Training logs
├── src/
│   ├── data_utils.py          - Data loading functions
│   ├── dataset.py             - Custom dataset class
│   └── model_utils.py         - Model loading utilities
├── train.py                   - Training script
├── predict.py                 - Inference script
├── requirements.txt           - Python dependencies
└── README.md

---

## How to Train the Model

1. Place your training data at data/train.csv.  
   The file must include two columns:
   - text: The review sentence  
   - label: 0 = negative, 1 = positive

2. Run the training script:

    python train.py

3. After training:
   - The model and tokenizer will be saved under the model/ directory.
   - Logs will be saved under outputs/logs/.

### Download Trained Model (MEGA)

Due to GitHub's file size limit, the model is not included in the repository.  
You can download it from the following link and place it under the `model/` folder:
[Download model.zip from MEGA](https://mega.nz/file/tlA0DYIY#x0IIChJS-BchN7HdARZBJwkyfWlgVJkGGKIKkwyjeZ4)

---

## How to Predict Sentiment

Once the trained model is available under the model/ folder, run:

    python predict.py

You will see outputs like:

    'This movie was amazing!' → positive  
    'I hated this film.' → negative

You may customize predict.py to accept user input or test batch files.

---

## Installation

To install the necessary dependencies:

    pip install -r requirements.txt

Contents of requirements.txt:

    transformers
    torch
    scikit-learn
    pandas

---

## Model Details

This project fine-tunes the distilbert-base-uncased model using Hugging Face Transformers and PyTorch.

- Tokenizer: DistilBertTokenizer
- Model: DistilBertForSequenceClassification
- Max input length: 512 tokens
- Batch size: 16
- Epochs: 2
- Output directory: model/
- Logs: outputs/logs/

---
---

## Evaluation Results

The model was evaluated on a 20% split from the original training data.  
Here are the performance metrics based on that validation set:

| Metric     | Score  |
|------------|--------|
| Accuracy   | 95.00% |
| F1 Score   | 94.12% |
| Precision  | 96.39% |
| Recall     | 91.95% |

These results indicate that the model is not only highly accurate but also balanced in terms of false positives and false negatives. It performs well in both detecting positive and negative sentiment accurately.


## References

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding: https://arxiv.org/abs/1810.04805
- DistilBERT: A distilled version of BERT: https://arxiv.org/abs/1910.01108
- IMDb Dataset on Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Stanford NLP - Sentiment Analysis: https://nlp.stanford.edu/sentiment/