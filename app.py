import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from Model.CNN import MemeCNN

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = MemeCNN()
model.load_state_dict(torch.load("meme_model.pt", map_location=device))
model.to(device)
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# UI
st.title("🔥 Meme Vulgarity Detector")

uploaded_file = st.file_uploader("Upload Meme Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        st.error(f"❌ Vulgar Meme (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Non-Vulgar Meme (Confidence: {1-prob:.2f})")