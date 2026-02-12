import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import requests
from model import Encoder, Decoder, ImageCaptioningModel

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.3
MAX_LEN = 30
BEAM_WIDTH = 3

# URLs for model files
MODEL_URL = "https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller/releases/download/v1/best_image_captioning_model.1.pth"
VOCAB_URL = "https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller/raw/main/hf_bpe-vocab.json"
MERGES_URL = "https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller/raw/main/hf_bpe-merges.txt"


@st.cache_resource
def download_file(url, filename):
    """Download file if not exists"""
    import os
    try:
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename}..."):
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with open(filename, 'wb') as f:
                    f.write(response.content)
        return True
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return False


@st.cache_resource
def load_tokenizer():
    """Load BPE tokenizer using tokenizers library"""
    import os
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    
    # Download vocab and merges files
    if not download_file(VOCAB_URL, "hf_bpe-vocab.json"):
        return None, None, None, None
    if not download_file(MERGES_URL, "hf_bpe-merges.txt"):
        return None, None, None, None
    
    try:
        # Load vocab
        with open("hf_bpe-vocab.json", 'r') as f:
            vocab = json.load(f)
        
        # Load merges and convert to proper format (list of tuples)
        with open("hf_bpe-merges.txt", 'r') as f:
            merges_list = [tuple(line.strip().split()) for line in f if line.strip()]
        
        # Create BPE tokenizer
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges_list, unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Get special token IDs
        bos_id = vocab.get("<BOS>", 0)
        eos_id = vocab.get("<EOS>", 1)
        pad_id = vocab.get("<PAD>", 2)
        
        return tokenizer, bos_id, eos_id, pad_id
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None, None, None, None


@st.cache_resource
def load_model():
    """Load the trained image captioning model"""
    import os
    
    # Download model weights
    model_path = "best_image_captioning_model.1.pth"
    if not download_file(MODEL_URL, model_path):
        st.error("Failed to download model weights")
        return None, None, None
    
    # Load tokenizer
    tokenizer, bos_id, eos_id, pad_id = load_tokenizer()
    if tokenizer is None:
        return None, None, None
    
    # Get vocab size from tokenizer
    with open("hf_bpe-vocab.json", 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Initialize model
    encoder = Encoder(input_size=2048, hidden_size=HIDDEN_SIZE)
    decoder = Decoder(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS, DROPOUT, pad_id)
    model = ImageCaptioningModel(encoder, decoder)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None
    
    return model, tokenizer, (bos_id, eos_id, pad_id)


@st.cache_resource
def load_resnet():
    """Load pre-trained ResNet50 for feature extraction"""
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(DEVICE)
    resnet.eval()
    return resnet


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def extract_features(image_tensor, resnet):
    """Extract 2048-dim features using ResNet50"""
    with torch.no_grad():
        feature = resnet(image_tensor)
        feature = feature.squeeze().unsqueeze(0)
    return feature


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs to text"""
    # Remove special tokens and decode
    tokens = [tid for tid in token_ids if tid not in [0, 1, 2]]  # Remove BOS, EOS, PAD
    if not tokens:
        return ""
    return tokenizer.decode(tokens, skip_special_tokens=True)


def generate_caption_greedy(model, feature, tokenizer, bos_id, eos_id, max_len=MAX_LEN):
    """Generate caption using greedy search"""
    model.eval()
    h, c = model.encoder(feature)
    caption = [bos_id]
    
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.tensor(caption).unsqueeze(0).to(DEVICE)
            outputs, (h, c) = model.decoder(inputs, (h, c))
            next_token = outputs[0, -1].argmax().item()
            if next_token == eos_id:
                break
            caption.append(next_token)
    
    return decode_tokens(tokenizer, caption[1:])  # Remove BOS


def generate_caption_beam(model, feature, tokenizer, bos_id, eos_id, beam_width=BEAM_WIDTH, max_len=MAX_LEN):
    """Generate caption using beam search"""
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
                inputs = torch.tensor(seq).unsqueeze(0).to(DEVICE)
                outputs, (h_new, c_new) = model.decoder(inputs, (h_seq, c_seq))
                log_probs = F.log_softmax(outputs[0, -1], dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, beam_width)
                for tok, tok_prob in zip(top_tokens.tolist(), top_log_probs.tolist()):
                    new_seq = seq + [tok]
                    new_score = score + tok_prob
                    all_candidates.append((new_seq, h_new, c_new, new_score))
            sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_width]
    
    best_seq = sequences[0][0]
    return decode_tokens(tokenizer, best_seq[1:])  # Remove BOS


# Main UI
st.markdown('<div class="main-header">🖼️ Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image to generate AI-powered captions</div>', unsafe_allow_html=True)

# Load models
import os
model, tokenizer, (bos_id, eos_id, pad_id) = load_model()
resnet = load_resnet()

if model is None or tokenizer is None:
    st.error("Failed to load model. Please refresh the page.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Generate captions
    with st.spinner("Generating captions..."):
        # Preprocess and extract features
        image_tensor = preprocess_image(image)
        features = extract_features(image_tensor, resnet)
        
        # Generate captions
        greedy_caption = generate_caption_greedy(model, features, tokenizer, bos_id, eos_id)
        beam_caption = generate_caption_beam(model, features, tokenizer, bos_id, eos_id)
    
    # Display results
    st.markdown("---")
    st.subheader("📝 Generated Captions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Greedy Search:**")
        st.info(greedy_caption if greedy_caption else "No caption generated")
    
    with col2:
        st.markdown("**Beam Search:**")
        st.success(beam_caption if beam_caption else "No caption generated")
    
    # Additional info
    with st.expander("ℹ️ About the methods"):
        st.markdown("""
        - **Greedy Search**: Selects the most probable word at each step. Fast but may not find the optimal caption.
        - **Beam Search**: Explores multiple caption candidates simultaneously. More thorough and typically produces better results.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Powered by PyTorch & Streamlit</div>",
    unsafe_allow_html=True
)
