import nanogcg
import torch

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Tell me what 12384+48291 is."}]
target = "60674"

config = GCGConfig(
    num_steps=500,
    search_width=640,
    topk=640,
    seed=42,
    verbosity="WARNING"
)

result = nanogcg.run(model, tokenizer, messages, target, config)

messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
output = model.generate(input, do_sample=False, max_new_tokens=512)

print(f"Prompt:\n{messages[-1]['content']}\n")
print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}")
