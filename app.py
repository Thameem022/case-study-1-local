import os
import torch
import gradio as gr
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- Auth & model selection --------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. In Spaces, add it under Settings → Repository secrets.")

login(token=HF_TOKEN)

MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load tokenizer & model --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float32,   # CPU-safe; switch to bfloat16/float16 for GPU if desired
)
model.to(DEVICE)
model.eval()

# -------- Ensure chat template exists (Llama 3 format) --------
if tokenizer.chat_template is None:
    tokenizer.chat_template = """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""

# Prefer the explicit end-of-turn token; fall back to eos if needed
def _get_eot_id(tok):
    tid = tok.convert_tokens_to_ids("<|eot_id|>")
    return tid if isinstance(tid, int) and tid >= 0 else tok.eos_token_id

EOT_ID = _get_eot_id(tokenizer)
PAD_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

SYSTEM_PROMPT = (
    "You are Sustainable.ai, a friendly, encouraging, and knowledgeable assistant. "
    "Your sole purpose is to offer simple, practical, sustainable alternatives to everyday actions. "
    "Be supportive and non-judgmental."
)

@torch.inference_mode()
def chat(message, history):
    """
    Gradio ChatInterface signature:
      - message: str (latest user turn)
      - history: list[[user, assistant], ...]
    Returns a single string.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=EOT_ID,
        pad_token_id=PAD_ID,
    )

    # Decode only the newly generated portion
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text

# -------- Define the app at module scope (important for Spaces) --------
demo = gr.ChatInterface(
    fn=chat,
    title="Sustainable.ai 🌿",
    description="Tell me what you plan to do, and I’ll suggest a simpler, greener alternative.",
    submit_btn="Suggest",
    retry_btn="Regenerate",
    clear_btn="Clear",
)
# Optional: enable queue with a reasonable concurrency
demo = demo.queue(max_size=32, concurrency_count=2)

# Local dev
if __name__ == "__main__":
    demo.launch()
