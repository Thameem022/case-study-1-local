import os
import torch
import gradio as gr
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========================== Setup ==========================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. In Spaces, add it under Settings → Repository secrets.")

login(token=HF_TOKEN)

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-270m-it")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# If pad is missing, map to eos
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
).to(DEVICE)
model.eval()

# Use model's provided chat template if present; otherwise a minimal one.
if tokenizer.chat_template is None:
    tokenizer.chat_template = """{% for message in messages -%}
<start_of_turn>{{ message['role'] }}
{{ message['content'] }}<end_of_turn>
{% endfor -%}{% if add_generation_prompt %}<start_of_turn>assistant
{% endif %}"""

EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (EOS_ID or 0)

# Detect which assistant role the template expects.
# Many Gemma-3 templates use "assistant"; some forks use "model".
TEMPLATE_STR = tokenizer.chat_template or ""
ASSISTANT_ROLE = "assistant" if "assistant" in TEMPLATE_STR else "model"

# ================== Sustainability Logic ===================
EMISSIONS_FACTORS = {
    "transportation": {"car": 2.3, "bus": 0.1, "train": 0.04, "plane": 0.25},   # kg CO2 per km
    "food": {"meat": 6.0, "vegetarian": 1.5, "vegan": 1.0},                    # kg CO2 per meal
}

def calculate_footprint(car_km, bus_km, train_km, air_km_week, meat_meals, vegetarian_meals, vegan_meals):
    transport_emissions = (
        car_km * EMISSIONS_FACTORS["transportation"]["car"] +
        bus_km * EMISSIONS_FACTORS["transportation"]["bus"] +
        train_km * EMISSIONS_FACTORS["transportation"]["train"] +
        air_km_week * EMISSIONS_FACTORS["transportation"]["plane"]
    )
    food_emissions = (
        meat_meals * EMISSIONS_FACTORS["food"]["meat"] +
        vegetarian_meals * EMISSIONS_FACTORS["food"]["vegetarian"] +
        vegan_meals * EMISSIONS_FACTORS["food"]["vegan"]
    )
    total_emissions = transport_emissions + food_emissions
    stats = {
        "trees": round(total_emissions / 21),     # playful rough equivalents
        "flights": round(total_emissions / 500),
        "driving100km": round(total_emissions / 230),
    }
    return total_emissions, stats

GUIDANCE = (
    "You are Sustainable.ai. Give practical, encouraging sustainability alternatives only.\n"
    "Constraints:\n"
    "1) Reply in 3 to 6 short bullet points.\n"
    "2) Include a rough CO2 saving per bullet.\n"
    "3) No moralizing.\n"
    "4) Offer 1 easy switch, 1 medium switch, 1 stretch goal.\n"
)

GEN_KW = dict(
    max_new_tokens=256,
    do_sample=False,            # deterministic for stability
    temperature=0.0,
    repetition_penalty=1.05,
    eos_token_id=EOS_ID,
    pad_token_id=PAD_ID,
)

# ======================= Utilities ========================
def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _add(conv, role, content):
    """Append a role/content pair if content is non-empty."""
    if not content:
        return
    # Map roles to the template's expected assistant role
    if role == "assistant":
        role = ASSISTANT_ROLE
    elif role == "system":
        # Gemma templates often do not support 'system'; treat as user context
        role = "user"
    elif role not in ("user", "assistant", "model"):
        role = "user"
    conv.append({"role": role, "content": str(content)})

def _normalize_from_history(history, conv):
    """
    history may be:
      - list[tuple(user, assistant)]
      - list[dict(role, content)]
    """
    if not isinstance(history, list):
        return
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            u, a = item
            if u:
                _add(conv, "user", u)
            if a:
                _add(conv, "assistant", a)
        elif isinstance(item, dict):
            _add(conv, item.get("role", "user"), item.get("content", ""))

def _normalize_from_messages(messages, conv):
    """
    messages may be:
      - list[dict(role, content)]
      - list[str]
      - str
      - None
    """
    if messages is None:
        return
    if isinstance(messages, list):
        # If dicts, use them; if strings, treat each as a user turn
        for m in messages:
            if isinstance(m, dict):
                _add(conv, m.get("role", "user"), m.get("content", ""))
            elif isinstance(m, str):
                _add(conv, "user", m)
    elif isinstance(messages, str):
        _add(conv, "user", messages)

def _merge_consecutive_same_role(conv):
    """Merge consecutive same-role messages to satisfy strict alternation."""
    if not conv:
        return conv
    merged = [conv[0]]
    for msg in conv[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] = (merged[-1]["content"].rstrip() + "\n\n" + msg["content"].lstrip())
        else:
            merged.append(msg)
    return merged

def _ensure_last_is_user(conv):
    """
    For add_generation_prompt=True, the template expects the last message to be a user turn.
    If the last is assistant/model, append a light user nudge.
    """
    if not conv:
        return [{"role": "user", "content": "Please respond."}]
    last_role = conv[-1]["role"]
    if last_role in ("assistant", "model"):
        conv.append({"role": "user", "content": "Continue."})
    return conv

# ===================== Chat Function ======================
# Be tolerant to Gradio shapes: (messages, history, ...) or (message, history, ...)
@torch.inference_mode()
def chat(messages=None, history=None, car_km=0, bus_km=0, train_km=0, air_km_month=0, meat_meals=0, vegetarian_meals=0, vegan_meals=0, *args):
    # Convert monthly air travel to weekly to keep units consistent
    air_km_week = _to_float(air_km_month) / 4.3

    footprint, stats = calculate_footprint(
        _to_float(car_km), _to_float(bus_km), _to_float(train_km), air_km_week,
        _to_float(meat_meals), _to_float(vegetarian_meals), _to_float(vegan_meals)
    )

    context = (
        f"User’s estimated weekly footprint: {footprint:.1f} kg CO2.\n"
        f"Equivalents: about {stats['trees']} trees or {stats['flights']} short flights.\n"
        "Help them lower this number."
    )

    # Build conversation seed with guidance folded into the FIRST user turn.
    conv = []

    # Prefer Gradio messages if they are structured; otherwise use history.
    # We'll assemble a provisional conv, then fold guidance in.
    provisional = []
    _normalize_from_history(history, provisional)
    _normalize_from_messages(messages, provisional)

    # If first message exists and is a user turn, prepend guidance+context to that same message.
    guidance_block = GUIDANCE + "\n" + context
    if provisional and provisional[0]["role"] == "user":
        provisional[0]["content"] = guidance_block + "\n\n" + provisional[0]["content"]
    else:
        # Start with a user turn containing guidance and context
        provisional.insert(0, {"role": "user", "content": guidance_block})

    # Merge consecutive same-role messages to satisfy alternation
    conv = _merge_consecutive_same_role(provisional)

    # Ensure final message is a user turn for add_generation_prompt=True
    conv = _ensure_last_is_user(conv)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    outputs = model.generate(**inputs, **GEN_KW)
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Light formatting nudge toward bullets
    if not any(ch in text for ch in ("•", "-", "*")):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            text = "\n".join(f"• {l}" for l in lines[:6])

    return text

# ========================== UI ============================
with gr.Blocks(css="""
    body {
        background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
        font-family: 'Inter', sans-serif;
    }
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title-text {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 16px;
        color: #444;
        margin-bottom: 30px;
    }
    footer {
        text-align: center;
        font-size: 12px;
        color: #666;
        margin-top: 20px;
    }
""") as demo:
    with gr.Column():
        gr.HTML("<div class='title-text'>🌍 Eco Wise AI</div>")
        gr.HTML("<div class='subtitle-text'>Track your weekly habits and get personalized sustainability tips 🌱</div>")

        with gr.Row():
            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 🚗 Transportation (per week)")
                car_input = gr.Number(label="🚘 Car Travel (km)", value=0)
                bus_input = gr.Number(label="🚌 Bus Travel (km)", value=0)
                train_input = gr.Number(label="🚆 Train Travel (km)", value=0)
                air_input = gr.Number(label="✈️ Air Travel (km/month)", value=0)

            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 🍽️ Food Habits (per week)")
                meat_input = gr.Number(label="🥩 Meat Meals", value=0)
                vegetarian_input = gr.Number(label="🥗 Vegetarian Meals", value=0)
                vegan_input = gr.Number(label="🌱 Vegan Meals", value=0)

        with gr.Group(elem_classes="section-card"):
            gr.Markdown("### 💬 Chat with Sustainable.ai")
            chatbot = gr.ChatInterface(
                fn=chat,
                type="messages",  # role/content dicts when available
                additional_inputs=[
                    car_input, bus_input, train_input, air_input,
                    meat_input, vegetarian_input, vegan_input
                ],
            )

        gr.HTML("<footer>⚡ Built with Gemma 3 270M IT & Gradio • Eco Wise AI © 2025</footer>")

# Queue with concurrency control
demo = demo.queue(max_size=32, default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_port=8080)
