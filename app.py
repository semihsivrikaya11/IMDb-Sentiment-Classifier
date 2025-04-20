import gradio as gr
from src.model_utils import load_model
from transformers import DistilBertTokenizer
import torch

# Load model and tokenizer
tokenizer, model, device = load_model("model")

# Prediction function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        prediction = torch.argmax(output.logits, dim=1).item()
        return "positive" if prediction == 1 else "negative"

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review..."),
    outputs="text",
    title="IMDb Sentiment Classifier",
    description="Enter a movie review and see if it's classified as positive or negative."
)

if __name__ == "__main__":
    demo.launch()
