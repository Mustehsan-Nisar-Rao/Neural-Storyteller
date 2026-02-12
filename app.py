import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import os
import tempfile
from model import ImageCaptioningModel, Encoder, Decoder
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
import io

# ------------------- Configuration -------------------
@st.cache_resource
def load_config():
    """Load configuration and special tokens"""
    config = {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'dropout': 0.3,
        'max_len': 30,
        'beam_width': 3,
        'bos_id': 1,  # BOS token ID (adjust based on your tokenizer)
        'eos_id': 2,  # EOS token ID (adjust based on your tokenizer)
        'pad_id': 0,  # PAD token ID
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    return config

@st.cache_resource
def load_tokenizer():
    """Download and load BPE tokenizer from GitHub"""
    try:
        vocab_url = "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Neural-Storyteller/main/hf_bpe-vocab.json"
        merges_url = "https://raw.githubusercontent.com/Mustehsan-Nisar-Rao/Neural-Storyteller/main/hf_bpe-merges.txt"
        
        # Download vocab and merges
        vocab_response = requests.get(vocab_url)
        merges_response = requests.get(merges_url)
        
        # Save to temporary files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as vocab_file:
            vocab_file.write(vocab_response.text)
            vocab_path = vocab_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as merges_file:
            merges_file.write(merges_response.text)
            merges_path = merges_file.name
        
        # Create tokenizer
        tokenizer = Tokenizer(BPE.from_file(vocab=vocab_path, merges=merges_path))
        tokenizer.pre_tokenizer = ByteLevel()
        
        # Clean up temp files
        os.unlink(vocab_path)
        os.unlink(merges_path)
        
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

@st.cache_resource
def load_model():
    """Download model weights and create model instance"""
    config = load_config()
    model_url = "https://github.com/Mustehsan-Nisar-Rao/Neural-Storyteller/releases/download/v1/best_image_captioning_model.1.pth"
    
    with st.spinner("Downloading model weights..."):
        try:
            # Download model weights
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                model_path = tmp_file.name
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Get vocabulary size from checkpoint
            decoder_state_dict = {k.replace('decoder.', ''): v 
                                for k, v in checkpoint['state_dict'].items() 
                                if k.startswith('decoder.')}
            vocab_size = decoder_state_dict['embed.weight'].shape[0]
            
            # Create model
            encoder = Encoder(input_size=2048, hidden_size=config['hidden_size'])
            decoder = Decoder(
                embed_size=config['embed_size'],
                hidden_size=config['hidden_size'],
                vocab_size=vocab_size,
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                pad_id=config['pad_id']
            )
            
            model = ImageCaptioningModel(encoder, decoder)
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
            model.to(config['device'])
            model.eval()
            
            # Clean up temp file
            os.unlink(model_path)
            
            return model, vocab_size
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

@st.cache_resource
def load_resnet():
    """Load pretrained ResNet50 for feature extraction"""
    config = load_config()
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
    resnet = resnet.to(config['device'])
    resnet.eval()
    return resnet

# ------------------- Image Transform -------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# ------------------- Caption Generation -------------------
def generate_caption_greedy(model, feature, tokenizer, config, max_len=30):
    """Generate caption using greedy search"""
    model.eval()
    h, c = model.encoder(feature)
    caption = [config['bos_id']]
    
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.tensor(caption).unsqueeze(0).to(config['device'])
            outputs, (h, c) = model.decoder(inputs, (h, c))
            next_token = outputs[0, -1].argmax().item()
            if next_token == config['eos_id']:
                break
            caption.append(next_token)
    
    # Decode caption
    try:
        decoded = tokenizer.decode(caption[1:])  # Remove BOS
    except:
        decoded = " ".join([str(idx) for idx in caption[1:]])
    
    return decoded

def generate_caption_beam(model, feature, tokenizer, config, beam_width=3, max_len=30):
    """Generate caption using beam search"""
    model.eval()
    h, c = model.encoder(feature)
    sequences = [([config['bos_id']], h, c, 0.0)]
    
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            for seq, h_seq, c_seq, score in sequences:
                if seq[-1] == config['eos_id']:
                    all_candidates.append((seq, h_seq, c_seq, score))
                    continue
                inputs = torch.tensor(seq).unsqueeze(0).to(config['device'])
                outputs, (h_new, c_new) = model.decoder(inputs, (h_seq, c_seq))
                log_probs = F.log_softmax(outputs[0, -1], dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, beam_width)
                
                for tok, tok_prob in zip(top_tokens.tolist(), top_log_probs.tolist()):
                    new_seq = seq + [tok]
                    new_score = score + tok_prob
                    all_candidates.append((new_seq, h_new, c_new, new_score))
            
            sequences = sorted(all_candidates, key=lambda x: x[3], reverse=True)[:beam_width]
    
    best_seq = sequences[0][0]
    
    # Decode caption
    try:
        decoded = tokenizer.decode(best_seq[1:])  # Remove BOS
    except:
        decoded = " ".join([str(idx) for idx in best_seq[1:]])
    
    return decoded

# ------------------- Streamlit UI -------------------
def main():
    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon="📷",
        layout="wide"
    )
    
    st.title("📷 Neural Storyteller - Image Caption Generator")
    st.markdown("Upload an image and get AI-generated captions using **Greedy** and **Beam Search** decoding.")
    
    # Load all resources
    with st.spinner("Loading model and tokenizer... Please wait."):
        config = load_config()
        tokenizer = load_tokenizer()
        model, vocab_size = load_model()
        resnet = load_resnet()
        transform = get_transform()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model or tokenizer. Please check your internet connection and try again.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        beam_width = st.slider("Beam Width", min_value=1, max_value=5, value=3)
        max_length = st.slider("Max Caption Length", min_value=20, max_value=50, value=30)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write(f"**Vocabulary Size:** {vocab_size}")
        st.write(f"**Device:** {config['device']}")
        st.write(f"**Hidden Size:** {config['hidden_size']}")
        st.write(f"**Embed Size:** {config['embed_size']}")
        
        st.markdown("---")
        st.header("📌 How to use")
        st.write("1. Upload an image (JPG, JPEG, PNG)")
        st.write("2. Click 'Generate Caption'")
        st.write("3. View results")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to generate caption"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("📝 Generated Captions")
        
        if uploaded_file is not None and st.button("🚀 Generate Caption", type="primary"):
            try:
                with st.spinner("Generating captions..."):
                    # Process image
                    img_tensor = transform(image).unsqueeze(0).to(config['device'])
                    
                    # Extract features
                    with torch.no_grad():
                        feature = resnet(img_tensor)
                        feature = feature.squeeze().unsqueeze(0)  # [1, 2048]
                    
                    # Generate captions
                    greedy_caption = generate_caption_greedy(
                        model, feature, tokenizer, config, max_len=max_length
                    )
                    
                    beam_caption = generate_caption_beam(
                        model, feature, tokenizer, config, 
                        beam_width=beam_width, max_len=max_length
                    )
                    
                    # Display results in nice boxes
                    st.success("✅ Captions generated successfully!")
                    
                    # Greedy caption
                    st.markdown("### 🎯 Greedy Search")
                    st.info(greedy_caption)
                    
                    # Beam search caption
                    st.markdown(f"### 🔍 Beam Search (width={beam_width})")
                    st.success(beam_caption)
                    
                    # Add download buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.download_button(
                            label="📥 Download Greedy Caption",
                            data=greedy_caption,
                            file_name="greedy_caption.txt",
                            mime="text/plain"
                        )
                    with col_b:
                        st.download_button(
                            label="📥 Download Beam Caption",
                            data=beam_caption,
                            file_name="beam_caption.txt",
                            mime="text/plain"
                        )
                        
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                st.exception(e)
        elif uploaded_file is None:
            st.info("👆 Please upload an image first")
        else:
            st.info("Click 'Generate Caption' to see results")

if __name__ == "__main__":
    main()
