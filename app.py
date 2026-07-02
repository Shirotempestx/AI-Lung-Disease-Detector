import gradio as gr
from ultralytics import YOLO

# Load your actual trained model!
model = YOLO('best.pt')

def predict_xray(image):
    # Run the real image through your trained AI
    results = model(image)
    
    # Extract the probabilities for the 3 classes and send them to the interface
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    return {names_dict[i]: probs[i] for i in range(len(names_dict))}

app = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="⚕️ AI Lung Disease Detector",
    description="Upload a chest X-Ray (Radiography) to instantly detect Normal, Pneumonia, or Tuberculosis classifications."
)

app.launch()