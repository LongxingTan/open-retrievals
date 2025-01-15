import torch
from transformers import AutoModel, AutoTokenizer
from mteb import MTEB

# Step 1: Load the fine-tuned Nomic model and tokenizer
model_name = "/root/workspace/example_data/output"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class CustomEmbedder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, texts, batch_size=32, **kwargs):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Extract the CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).numpy()
embedder = CustomEmbedder(model=model, tokenizer=tokenizer, device=device)

# # Step 2: Define a function to encode texts using the model
# def encode(texts):
#     # Tokenize the input texts
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
#     # Move inputs to the same device as the model
#     inputs = {key: value.to(device) for key, value in inputs.items()}
    
#     # Get the embeddings from the model
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Use the [CLS] token's embedding as the sentence embedding
#     embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    
#     return embeddings

# Step 3: Load the MTEB benchmark and specify the task
task = "STS12"
benchmark = MTEB(tasks=[task])

# Step 5: Run the benchmark on the selected task
print(f"Running benchmark on task: {task}")
evaluation = benchmark.run(model=embedder)  # Pass only the model encoding function

# Step 6: Print the results
print(f"Results for {task}:")
print(evaluation)

# Optionally, save the results to a file
with open("mteb_results_sts12.txt", "w") as f:
    f.write(f"Results for {task}:\n")
    f.write(str(evaluation))