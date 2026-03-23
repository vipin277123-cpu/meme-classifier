import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# Dataset Class
class MemeDataset(Dataset):

    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = self.data.iloc[idx,0]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB",(64,64))

        label_text = str(self.data.iloc[idx,3]).lower().strip()

        label = 1 if label_text == "vulgar" else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# CNN Model
class MemeCNN(nn.Module):

    def __init__(self):
        super(MemeCNN,self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32*16*16,128)
        self.fc2 = nn.Linear(128,1)

    def forward(self,x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()