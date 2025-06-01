import os

cache_location = "/scratch/qido00001/tuning/.cache"
os.environ["HF_HOME"] = cache_location
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_location

import transformers
import torch
from huggingface_hub import login

if __name__ == "__main__":
    with open("/scratch/qido00001/hf_key.txt", "r") as file:
        key = file.readline()
    login(key)

    model_id = "meta-llama/Meta-Llama-3-8B-instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=torch.cuda.current_device(),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a software engineer who helps me write python code.",
        },
        {
            "role": "user",
            "content": "I need a function that takes a list of numbers and returns the sum of the numbers.",
        },
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    result = pipeline(
        prompt,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )

    answer = result[0]["generated_text"][len(prompt) :]
    print(answer)
