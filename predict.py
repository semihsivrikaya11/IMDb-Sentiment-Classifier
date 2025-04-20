from src.model_utils import load_model
from transformers import DistilBertTokenizer
import torch

# Load model and tokenizer
tokenizer, model, device = load_model("model")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
    return "positive" if predicted == 1 else "negative"

# Example usage
example1 = "This movie was fantastic!"
example2 = "The plot was boring and predictable."

print(f"'{example1}' → {predict_sentiment(example1)}")
print(f"'{example2}' → {predict_sentiment(example2)}")