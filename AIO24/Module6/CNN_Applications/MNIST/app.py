import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

class LeNetClassifier(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels = 1, out_channels = 6, kernel_size = 5,
        padding = 'same'
    )
    self.avgpool1 = nn.AvgPool2d(kernel_size = 2)
    self.conv2 = nn.Conv2d(
        in_channels = 6, out_channels = 16, kernel_size = 5
    )
    self.avgpool2 = nn.AvgPool2d(kernel_size = 2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
    self.fc2 = nn.Linear(in_features = 120, out_features = 84)
    self.fc3 = nn.Linear(in_features = 84, out_features = num_classes)

  def forward(self, x):
    outputs = self.conv1(x)
    outputs = self.avgpool1(outputs)
    outputs = F.relu(outputs)
    outputs = self.conv2(outputs)
    outputs = self.avgpool2(outputs)
    outputs = F.relu(outputs)
    outputs = self.flatten(outputs)
    outputs = self.fc1(outputs)
    outputs = self.fc2(outputs)
    outputs = self.fc3(outputs)
    return outputs

@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    lenet_model.eval()
    return lenet_model

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_file_directory, 'lenet_model.pt')
model = load_model(model_path)

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item()*100, yhat.item()

def main():
    st.title('Digit Recognition')
    st.subheader('Model: LeNet. Dataset: MNIST')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the digit {label} with {p:.2f} % probability.") 

    elif option == "Run Example Image":
        example_image_path = os.path.join(current_file_directory, 'demo_8.png')
        image = Image.open(example_image_path)
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The image is of the digit {label} with {p:.2f} % probability.") 

if __name__ == '__main__':
    main()