import torch

IMAGE_PLACEHOLDER_INDEX = -200
IGNORE_INDEX = -100
device = 'cuda:1'
# tokenizer_model_max_length = 11
tokenizer_model_max_length = 20

tokenizer_padding_side = 'right'


def embed_tokens(tensor, dim=3):
    return tensor.unsqueeze(-1).repeat(1, dim)


input_ids = torch.tensor([
    [1, 2, 3, 4, 5, 6, -200, 7, 8, 9, 10, 0, 0],
    [11, 22, 33, -200, 44, 55, 0, 0, 0, 0, 0, 0, 0]
], device=device, dtype=torch.long
)

labels = torch.tensor([
    [-100, -100, -100, -100, 5, 6, -100, 7, 8, 9, 10, -100, -100],
    [-100, -100, -100, -100, 44, 55, -100, -100, -100, -100, -100, -100, -100]
], device=device, dtype=torch.long
)

attention_mask = torch.tensor([
    [True, True, True, True, True, True, True, True, True, True, True, False, False],
    [True, True, True, True, True, True, False, False, False, False, False, False, False],
], device=device, dtype=torch.bool
)

image_features = torch.tensor([
    [[999, 999, 999],
     [999, 999, 999],
     [999, 999, 999], ],

    [[888, 888, 888],
     [888, 888, 888],
     [888, 888, 888], ],
]).to(device=device, dtype=torch.float32)

# seg1
input_ids = [
    cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
]
labels = [
    cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)
]
print('=' * 90)
print(input_ids)
print(labels)

# seg2
print()
new_input_embeds = []
new_labels = []
for batch_idx, cur_input_ids in enumerate(input_ids):
    print('-' * 90)
    num_images = (cur_input_ids == IMAGE_PLACEHOLDER_INDEX).sum()
    if num_images != 1:
        print(f'number of <image-placeholder> found in this batch_idx f{batch_idx} is not 1, Something error!')
        continue

    # [-1, 6, 11]
    # [-1, 3, 6]
    image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_PLACEHOLDER_INDEX)[0].tolist() + \
                          [cur_input_ids.shape[0]]
    print(f"image_token_indices: {image_token_indices}")

    cur_labels = labels[batch_idx]
    cur_input_ids_no_img = []
    cur_labels_no_img = []
    for i in range(len(image_token_indices) - 1):
        cur_input_ids_no_img.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
        cur_labels_no_img.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
    print(f'cur_input_ids_no_img: {cur_input_ids_no_img}')
    print(f'cur_labels_no_img: {cur_labels_no_img}')

    split_sizes = [x.shape[0] for x in cur_labels_no_img]
    print(f'split_sizes: {split_sizes}')
    print(f'torch.cat(cur_input_ids_no_img): {torch.cat(cur_input_ids_no_img)}')

    cur_input_embeds = embed_tokens(torch.cat(cur_input_ids_no_img))
    print(f"cur_input_embeds: {cur_input_embeds}")
    cur_input_embeds_no_img = torch.split(cur_input_embeds, split_sizes, dim=0)
    print(f"cur_input_embeds_no_img: {cur_input_embeds_no_img}")

    cur_new_input_embeds = []
    cur_new_labels = []

    cur_new_input_embeds.append(cur_input_embeds_no_img[0])
    cur_new_labels.append(cur_labels_no_img[0])
    cur_new_input_embeds.append(image_features[batch_idx])
    cur_new_labels.append(
        torch.full((image_features[batch_idx].shape[0],), IGNORE_INDEX,
                   device=cur_labels.device, dtype=cur_labels.dtype)
    )
    cur_new_input_embeds.append(cur_input_embeds_no_img[1])
    cur_new_labels.append(cur_labels_no_img[1])

    # print(f"cur_new_input_embeds: {cur_new_input_embeds}")
    # cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
    # print(f"cur_new_input_embeds: {cur_new_input_embeds}")

    cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    cur_new_labels = torch.cat(cur_new_labels)
    print(f"cur_new_input_embeds: {cur_new_input_embeds}")
    print(f"cur_new_labels: {cur_new_labels}")

    new_input_embeds.append(cur_new_input_embeds)
    new_labels.append(cur_new_labels)

print('=' * 90)
print(f"new_input_embeds: {new_input_embeds}")
print(f"new_labels: {new_labels}")

# seg3:
print()
if tokenizer_model_max_length is not None:
    new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

print('=' * 90)
print(f"new_input_embeds: {new_input_embeds}")
print(f"new_labels: {new_labels}")

# seg4
print()
max_len = max(x.shape[0] for x in new_input_embeds)
print(f'max_len: {max_len}')
batch_size = len(new_input_embeds)
print(f'batch_size: {batch_size}')

new_input_embeds_padded = []
new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX,
                               dtype=new_labels[0].dtype, device=new_labels[0].device)
attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=device)
position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
print(f'new_labels_padded: {new_labels_padded}')
print(f'attention_mask: {attention_mask}')
print(f'position_ids: {position_ids}')

for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    print('-' * 90)
    cur_len = cur_new_embed.shape[0]
    print(f'cur_len: {cur_len}')
    if tokenizer_padding_side == "left":
        new_input_embeds_padded.append(
            torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0)
        )
        if cur_len > 0:
            new_labels_padded[i, -cur_len:] = cur_new_labels
            attention_mask[i, -cur_len:] = True
            position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    else:
        new_input_embeds_padded.append(
            torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0)
        )
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
print(f'new_labels_padded: {new_labels_padded}')
print(f'new_input_embeds: {new_input_embeds}')
print(f'attention_mask: {attention_mask}')
print(f'position_ids: {position_ids}')
