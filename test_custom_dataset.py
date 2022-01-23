from utils.datasets_custom_pth import PthDataset
from torch.utils.data import DataLoader

training_data = PthDataset(folder_type="val")
train_loader = DataLoader(training_data)
pbar = enumerate(train_loader)
for i, (imgs, targets, paths, shapes) in pbar:
    print(shapes)
    break
