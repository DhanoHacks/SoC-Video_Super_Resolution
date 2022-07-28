# Train the model here
import torch
from torch import nn
from model import Net
from datasets import TrainDataset
import argparse

def train(scale, n, lr, d):
    # Device configuration
    device = torch.device(d)

    # Hyper-parameters 
    num_epochs = n
    batch_size = 4
    learning_rate = lr

    train_path = f"../training_image_database/91-image_x{scale}.h5" #change this to path to training dataset
    train_dataset=TrainDataset(train_path)
    MODEL_PATH=f"models/trained_model_x{scale}.pth"

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    model = Net().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            #forward pass
            sr = model(lr)
            loss = criterion(sr, hr)

            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100 == 0:
                print(f"Scale x{scale}, Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.7f}")
    print(f"Finished Training for Scale x{scale}")
    #saving the model parameters
    torch.save(model.state_dict(), MODEL_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--num_epochs', type=int, required=False, default=10)
parser.add_argument('--learning_rate', type=float, required=False, default=0.00001)
parser.add_argument('--device', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

train(args.scale, args.num_epochs, args.learning_rate, args.device)