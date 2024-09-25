import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Tell me what 12384+48291 is. underscore ####greSEAinement_pythonively source **** narrowed_plusaceousantly NSCoder \"\"\".irDecode Simply Reply Length"

models = [
        #"Qwen/Qwen2.5-0.5B-Instruct",
           "Qwen/Qwen2.5-1.5B-Instruct",
        #   "Qwen/Qwen2.5-3B-Instruct",
        #   "Qwen/Qwen2.5-7B-Instruct",
        #   "Qwen/Qwen2.5-14B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
         ]

device = 0

# Create a completion pipeline for each model and generate responses
for model_id in models:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [{"role": "user", "content": prompt}]

    input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    output = model.generate(input, do_sample=False, max_new_tokens=512)

    print(f"Model: {model_id}")
    print(f"Prompt:\n{messages[-1]['content']}\n")
    print(f"Generation:\n{tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]}")
