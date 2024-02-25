from Surface_Dataset import Surface_Dataset
import argparse
import torch
from UNet import UNet
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Running on "+ str(device))

transform = transforms.Compose([
    transforms.ToTensor()
])

def custom_loss_function(output, target, height, width):
    di = target - output
    n = (height *  width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str ,
        default="Convex_1Sphere_plane",
        help="Dir to the dataset consisting of images and depthMap (in .npz format)"
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=".png"
    )
    parser.add_argument(
        "--depthMap_suffix",
        type=str,
        default=".npz"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256
    )
    options = parser.parse_args()

    datset_path = options.dataset_path
    image_suffix = options.image_suffix
    depthMap_suffix = options.depthMap_suffix
    lr = options.lr
    batch_size = options.batch_size
    num_epochs = options.num_epochs
    height = options.height
    width = options.width

    ###################################################################################

    dataset = Surface_Dataset(datset_path, image_suffix, depthMap_suffix, transform) 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,)


    model = UNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
            for i, (images, depth_maps) in enumerate(t):
                images, depth_maps = images.to(device), depth_maps.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = criterion(outputs, depth_maps)
                # loss = custom_loss_function(outputs, depth_maps, height, width)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                # Print statistics
                running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')
                
    print('Finished Training')


Main()