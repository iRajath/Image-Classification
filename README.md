# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### **Step 1: Define the Objective**

Set the goal of creating a **Convolutional Neural Network (CNN)** capable of recognizing and categorizing handwritten digits from 0 through 9.

### **Step 2: Gather the Dataset**

Work with the MNIST dataset, which provides 60,000 images for training and 10,000 images for testing handwritten digit recognition models.

### **Step 3: Prepare the Data**

Transform the images into tensor format, scale pixel values to a normalized range to improve learning performance, and organize the data into DataLoaders to enable efficient batch-based processing.

### **Step 4: Build the Network Architecture**

Construct a CNN model that includes convolutional layers for extracting features, nonlinear activation functions such as ReLU, pooling layers to reduce spatial dimensions, and fully connected layers to perform final classification.

### **Step 5: Train the Network**

Train the model over several epochs using a suitable loss function like CrossEntropyLoss and an optimization algorithm such as Adam to adjust the model’s parameters.

### **Step 6: Assess Model Performance**

Evaluate the trained model using the test dataset. Calculate accuracy and examine performance further through a confusion matrix and a classification report.

### **Step 7: Save and Deploy the Model**

Store the trained model for later use, display prediction results for visualization, and integrate the model into an application if deployment is required.



## PROGRAM

### Name: S Rajath
### Register Number: 212224240127
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128*3*3,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)

  def forward(self,x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
```

```python
# Train the Model
def train_model(model,train_loader,num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print('Name: Abishek Priyan M')
    print('Register Number: 212224240004')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

<img width="265" height="162" alt="image" src="https://github.com/user-attachments/assets/ae1e1940-22c6-4c10-9999-40b90954b325" />

### Confusion Matrix

<img width="720" height="703" alt="image" src="https://github.com/user-attachments/assets/a29b2ebd-0a2e-4bb0-8fb0-5dce57a8bac3" />

### Classification Report

<img width="462" height="333" alt="image" src="https://github.com/user-attachments/assets/afccb944-6aae-4db4-a9c5-f7ad239acd59" />


### New Sample Data Prediction

<img width="410" height="494" alt="image" src="https://github.com/user-attachments/assets/1d97c8a9-c9d7-4f60-83df-d40159fe6b14" />

## RESULT
Include your result here.
