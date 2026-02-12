import streamlit as st
import os
import requests
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from model import Encoder, Decoder, ImageCaptioningModel
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# Download required files if not present
if not os.path.exists('hf_bpe-vocab.json'):
    url = 'https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Neural-Storyteller/main/hf_bpe-vocab.json'
    r = requests.get(url, allow_redirects=True)
    open('hf_bpe-vocab.json', 'wb').write(r.content)

if not os.path.exists('hf_bpe-merges.txt'):
    url = 'https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Neural-Storyteller/main/hf_bpe-merges.txt'
    r = requests.get(url, allow_redirects=True)
    open('hf_bpe-merges.txt', 'wb').write(r.content)

if not os.path.exists('best_image_captioning_model.1.pth'):
    url = 'https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller/releases/download/v1/best_image_captioning_model.1.pth'
    r = requests.get(url, allow_redirects=True)
    open('best_image_captioning_model.1.pth', 'wb').write(r.content)

# Load tokenizer
tokenizer = Tokenizer(models.BPE(vocab='hf_bpe-vocab.json', merges='hf_bpe-merges.txt'))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Special tokens with fallbacks to avoid errors
pad_id = tokenizer.token_to_id('<pad>') if tokenizer.token_to_id('<pad>') is not None else 0
bos_id = tokenizer.token_to_id('<s>') if tokenizer.token_to_id('<s>') is not None else 1
eos_id = tokenizer.token_to_id('</s>') if tokenizer.token_to_id('</s>') is not None else 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved file
loaded = torch.load('best_image_captioning_model.1.pth', map_location=device)

dropout = 0.3  # Default from provided code

if isinstance(loaded, dict):
    state_dict = loaded
    # Remove 'module.' prefix if present
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k.replace('module.', ''): k for k, v in state_dict.items()}
    # Infer hyperparameters from state_dict
    hidden_size = state_dict['encoder.fc.weight'].shape[0]
    embed_size = state_dict['decoder.embed.weight'].shape[1]
    vocab_size = state_dict['decoder.embed.weight'].shape[0]
    # Infer num_layers
    num_layers = 0
    while f'decoder.lstm.weight_ih_l{num_layers}' in state_dict:
        num_layers += 1
    # Create model
    encoder = Encoder(input_size=2048, hidden_size=hidden_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers, dropout=dropout, pad_id=pad_id).to(device)
    model = ImageCaptioningModel(encoder, decoder).to(device)
    model.load_state_dict(state_dict)
else:
    # Assume it's the full model
    model = loaded.to(device)
    # Infer hyperparameters from model (optional, but for consistency)
    hidden_size = model.encoder.fc.out_features
    embed_size = model.decoder.embed.embedding_dim
    vocab_size = model.decoder.embed.num_embeddings
    num_layers = model.decoder.lstm.num_layers
    dropout = model.decoder.dropout.p

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Pre-trained ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
resnet = resnet.to(device)
resnet.eval()

# Caption Generation Functions
def generate_caption_greedy(model, feature, max_len=30):
    model.eval()
    h, c = model.encoder(feature)
    caption = [bos_id]
   
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.tensor(caption).unsqueeze(0).to(device)
            outputs, (h, c) = model.decoder(inputs, (h, c))
            next_token = outputs[0, -1].argmax().item()
            if next_token == eos_id:
                break
            caption.append(next_token)
   
    return tokenizer.decode(caption[1:])  # remove BOS

def generate_caption_beam(model, feature, beam_width=3, max_len=30):
    model.eval()
    h, c = model.encoder(feature)
    sequences = [([bos_id], h, c, 0.0)]
   
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, h_seq, c_seq, score in sequences:
                if seq[-1] == eos_id:
                    all_candidates.append((seq, h_seq, c_seq, score))
                    continue
                inputs = torch.tensor(seq).unsqueeze(0).to(device)
                outputs, (h_new, c_new) = model.decoder(inputs, (h_seq, c_seq))
                log_probs = F.log_softmax(outputs[0, -1], dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, beam_width)
                for tok, tok_prob in zip(top_tokens.tolist(), top_log_probs.tolist()):
                    new_seq = seq + [tok]
                    new_score = score + tok_prob
                    all_candidates.append((new_seq, h_new, c_new, new_score))
            sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_width]
   
    best_seq = sequences[0][0]
    return tokenizer.decode(best_seq[1:])  # remove BOS

# Streamlit interface
st.title("Image Captioning App")
uploaded_file = st.file_uploader("Upload an image for captioning", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = resnet(img_tensor)  # [1, 2048, 1, 1]
        feature = feature.squeeze().unsqueeze(0)  # [1, 2048]
    
    greedy_caption = generate_caption_greedy(model, feature)
    beam_caption = generate_caption_beam(model, feature, beam_width=3)
    
    st.write("Generated Captions:")
    st.write("Greedy Caption:", greedy_caption)
    st.write("Beam Caption:", beam_caption)
