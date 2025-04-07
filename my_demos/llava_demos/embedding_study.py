import torch

device = 'cuda:1'
input_embeddings = torch.tensor(
    [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3],
     [4, 4, 4],
     [5, 5, 5]]
).to(device=device, dtype=torch.float32)

output_embeddings = torch.tensor(
    [[11, 11, 11, 11, 11],
     [22, 22, 22, 22, 22],
     [33, 33, 33, 33, 33]]
).to(device=device, dtype=torch.float32)

num_new_tokens = 1
input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    dim=0, keepdim=True)
output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    dim=0, keepdim=True)

print(f'input_embeddings_avg: {input_embeddings_avg}')
print(f'output_embeddings_avg: {output_embeddings_avg}')

input_embeddings[-num_new_tokens:] = input_embeddings_avg
output_embeddings[-num_new_tokens:] = output_embeddings_avg
print(f'input_embeddings_avg: {input_embeddings}')
print(f'output_embeddings_avg: {output_embeddings}')