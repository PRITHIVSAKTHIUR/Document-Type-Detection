import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Document-Type-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# ID to label mapping
id2label = {
    "0": "Advertisement-Doc",
    "1": "Hand-Written-Doc",
    "2": "Invoice-Doc",
    "3": "Letter-Doc",
    "4": "News-Article-Doc",
    "5": "Resume-Doc"
}

def detect_doc_type(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=detect_doc_type,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=6, label="Document Type"),
    title="Document-Type-Detection",
    description="Upload a document image to classify it as one of: Advertisement, Hand-Written, Invoice, Letter, News Article, or Resume."
)

if __name__ == "__main__":
    iface.launch()
