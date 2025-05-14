![Doc.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/iAFZ-Q4HW_F2KkL511tm8.png)

# **Document-Type-Detection**

> **Document-Type-Detection** is a multi-class image classification model based on `google/siglip2-base-patch16-224`, trained to detect and classify **types of documents** from scanned or photographed images. This model is helpful for **automated document sorting**, **OCR pipelines**, and **digital archiving systems**.

```py
Classification Report:
                   precision    recall  f1-score   support

Advertisement-Doc     0.8940    0.8940    0.8940      2000
 Hand-Written-Doc     0.9168    0.9310    0.9238      2000
      Invoice-Doc     0.9026    0.8940    0.8983      2000
       Letter-Doc     0.8380    0.8820    0.8594      2000
 News-Article-Doc     0.9258    0.8800    0.9023      2000
       Resume-Doc     0.9425    0.9340    0.9382      2000

         accuracy                         0.9025     12000
        macro avg     0.9033    0.9025    0.9027     12000
     weighted avg     0.9033    0.9025    0.9027     12000
```

![download (2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/pl1RVr-JTkI3hZLwHSQ0-.png)

---

## **Label Classes**

The model classifies images into the following document types:

```
0: Advertisement-Doc  
1: Hand-Written-Doc  
2: Invoice-Doc  
3: Letter-Doc  
4: News-Article-Doc  
5: Resume-Doc
```

---

## **Installation**

```bash
pip install transformers torch pillow gradio
```

---

## **Example Inference Code**

```python
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
```

---

## **Applications**

* **Automated Document Sorting**
* **Digital Libraries and Archives**
* **OCR Preprocessing**
* **Enterprise Document Management**
