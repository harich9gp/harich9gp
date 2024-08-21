# Harich9gp

```python
from mamba_model import Harich
from mamba_config import Harich
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Harich/Harich.7B")
input_text = 'A funny prompt would be'
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")["input_ids"].transpose(0,1)
model = MambaModel.from_pretrained(model_name = "Harich/Harich.7B").cuda().half()
tokens_to_generate = 20
model.eval()
with torch.no_grad():
    for _ in range(tokens_to_generate):
        out = model(input_ids)
        out_last = out[:, -1]
        idx = torch.argmax(out_last)[None, None]
        input_ids = torch.cat((input_ids, idx), dim=0)
input_ids = input_ids.transpose(0, 1)[0]
print(repr(tokenizer.decode(input_ids.cpu().numpy().tolist())))
```
