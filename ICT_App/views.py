from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Load the pre-trained model
model = resnet50(pretrained=True)
model.eval()

# Define the ImageNet classes
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    # Apply transformations to match the input dimensions and normalization of the pre-trained model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    img = preprocess(image)

    # Expand dimensions to match the input shape expected by the model
    img = torch.unsqueeze(img, 0)

    return img

def classify_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to the database
            new_image = UploadedImage(image=request.FILES['image'])
            new_image.save()

            # Preprocess the uploaded image
            image_path = new_image.image.path
            image = Image.open(image_path)
            preprocessed_image = preprocess_image(image)

            # Perform model inference
            with torch.no_grad():
                output = model(preprocessed_image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top_probabilities, top_indices = torch.topk(probabilities, k=5)

            # Get the classification labels and probabilities
            labels = [imagenet_classes[index.item()] for index in top_indices]
            probabilities = top_probabilities.tolist()

            # Set the classification result
            new_image.classification = list(zip(labels, probabilities))
            new_image.save()

            # Pass the classification result to the template for display
            context = {
                'form': form,
                'image': new_image,
                'classification': list(zip(labels, probabilities)),
            }
            return render(request, 'classify_image.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'classify_image.html', {'form': form})
