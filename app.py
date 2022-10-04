import sys
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

def generate(prompt, model, num_samples, steps=20, do_sample=True):
    tokeniser = BPETokenizer()
    x = tokeniser(prompt)
    x = x.expand(num_samples, -1)
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    for i in range(num_samples):
        out = tokeniser.decode(y[i].squeeze())
        return out

def main():
    print("Initialising model: General Purpose Transformer - 2")
    set_seed(3407)
    model = GPT.from_pretrained('gpt2')
    model.eval()
    num_samples = 1
    print("Ready.")
    while True:
        prompt = input(">> ")
        if prompt == "-1":
            break
        print("> Calculating response...")
        response = generate(prompt, model, num_samples)
        print(">", response)
    sys.exit(0)

if __name__ == "__main__":
    main()