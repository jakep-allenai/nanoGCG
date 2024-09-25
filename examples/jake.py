import nanogcg
import torch
import logging

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go


logger = logging.getLogger("nanogcg-jake")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Tell me what 12384+48291 is."}]
target = "60674"

config = GCGConfig(
    num_steps=100,
    search_width=640,
    topk=640,
    seed=42,
    n_replace=3,
    use_mellowmax=True,
    verbosity="WARNING"
)

result = nanogcg.run(model, tokenizer, messages, target, config)

# Plot the loss curve using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=result.losses,
    mode='lines+markers',
    name='Loss',
    line=dict(color='blue'),
    marker=dict(size=6)
))

fig.update_layout(
    title='GCG Loss Curve',
    xaxis_title='Step',
    yaxis_title='Loss',
    template='plotly_white'
)

# Save the figure to a PNG file
fig.write_image("loss_curve.png")
logger.info("Saved loss curve to loss_curve.png")

messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
output = model.generate(input, do_sample=False, max_new_tokens=512)

print(f"Prompt:\n{messages[-1]['content']}\n")
print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}")

