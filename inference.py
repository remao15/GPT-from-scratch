import torch
from languageModel import LanguageModel, encode, decode, device

torch.seed()

# build the model and load the trained weights
model = LanguageModel().to(device)
checkpoint = torch.load('model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


@torch.no_grad()
def generate(max_new_tokens=10000, prompt=""):
    """Sample text from the trained model; returns the decoded string."""
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens)[0].tolist()
    return decode(out)


if __name__ == '__main__':
    generated = generate(10000)
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(generated)
