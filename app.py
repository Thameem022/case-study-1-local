import gradio as gr
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# --- Step 1: Authentication and Model Loading ---
# Make sure you have set your HF_TOKEN in your environment (e.g., in Hugging Face Spaces secrets)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
login(token=hf_token)

model_id = "meta-llama/Llama-3.2-1B"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    token=hf_token, 
    torch_dtype=torch.float32 # Use float32 for CPU compatibility
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)


# --- Step 2: Manually Set Chat Template (The Fix) ---
# This block ensures the tokenizer has the correct Llama 3 chat format.
if tokenizer.chat_template is None:
    print("Chat template not found. Manually setting the Llama 3 template.")
    tokenizer.chat_template = """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""


# --- Step 3: System Prompt ---
SYSTEM_PROMPT = """You are Sustainable.ai, a friendly, encouraging, and knowledgeable AI assistant. Your sole purpose is to help users discover simple, practical, and sustainable alternatives to their everyday actions. You are a supportive guide on their eco-journey, never a critic. Your goal is to make sustainability feel accessible and effortless."""


# --- Step 4: The Chat Function ---
def chat(user_prompt):
    # Structure the conversation for the template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Use the tokenizer's chat template for correct formatting
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]

    # Generate a response, making sure it knows when to stop
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
    )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# --- Step 5: Gradio Interface ---
if __name__ == "__main__":
    demo = gr.Interface(
        fn=chat, 
        inputs=gr.Textbox(label="Your Action", placeholder="e.g., I'm driving to the store..."), 
        outputs=gr.Textbox(label="Sustainable Suggestion", lines=5), 
        title="Sustainable.ai 🌿",
        description="Tell me an action you're taking, and I'll suggest a simple, more sustainable alternative."
    )
    demo.launch()