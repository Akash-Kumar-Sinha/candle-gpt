from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

weights_dir = Path("../weights")
weights_dir.mkdir(parents=True, exist_ok=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")

for name, param in model.named_parameters():
    print(name, param.shape)
    values = param.data.numpy().flatten()
    with open(weights_dir / f"{name}.txt", "w") as f:
        f.write(' '.join(str(v) for v in values))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("../weights/tokenizer") 