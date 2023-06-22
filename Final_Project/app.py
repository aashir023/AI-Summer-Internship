import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the saved model
model = torch.load('vgg_model.pt', map_location=torch.device('cpu'))
model.eval()

# Define the class labels (if available)
class_labels = [
    'agricultural',
    'airplane',
    'baseballdiamond',
    'beach',
    'buildings',
    'chaparral',
    'denseresidential',
    'forest',
    'freeway',
    'golfcourse',
    'harbor',
    'intersection',
    'mediumresidential',
    'mobilehomepark',
    'overpass',
    'parkinglot',
    'river',
    'runway',
    'sparseresidential',
    'storagetanks',
    'tenniscourt'
]


def classify_image(image):
    # Preprocess the image
    preprocessed_image = transform(image).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    return predicted_class


def main():
    st.title("Image Classification Demo")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = image.convert("RGB")  # Ensure the image has 3 channels
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_class = classify_image(image)
        st.write("Predicted Class:", predicted_class)


if __name__ == '__main__':
    main()
