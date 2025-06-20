# app.py content for Hugging Face Space
# This code will be run by Hugging Face Spaces to launch your Gradio app.

import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch # Make sure torch is imported for device management

# Define your model ID (MUST MATCH the one you pushed to Hugging Face Hub in Phase 1!)
# IMPORTANT: REPLACE 'your-username/my-t5-tech-chatbot'

HF_MODEL_ID = "ikirezii/my-t5-tech-chatbot" 
MAX_TOKEN_LENGTH = 64 # Make sure this matches your training config from Colab

print(f"Loading model and tokenizer from Hugging Face Hub: {HF_MODEL_ID}")
# Load tokenizer and model directly from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID)

# Move model to GPU if available on the Space hardware, otherwise to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

# Define the chatbot inference function
def chatbot_inference(user_input):
    input_text = f"question: {user_input}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt", # Return PyTorch tensors
        max_length=MAX_TOKEN_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate response from the model
    output_sequences = model.generate(
        inputs["input_ids"],
        max_new_tokens=MAX_TOKEN_LENGTH,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    # Decode the generated token IDs back into human-readable text
    generated_response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_response

# Create the Gradio interface
# Note: You don't call iface.launch() here. Hugging Face Spaces handles that automatically.
iface = gr.Interface(
    fn=chatbot_inference,
    inputs=gr.Textbox(lines=2, placeholder="Type your tech support query here..."),
    outputs="text",
    title="Tech Support Chatbot (Fine-tuned T5-small - PyTorch)",
    description="Ask the chatbot a tech support question and get a generated response."
)

# Final line: This is what Hugging Face Spaces looks for to launch the app
iface.queue().launch() # Using queue and launch here is standard for Spaces deployment