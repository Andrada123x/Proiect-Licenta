import torch
from src.dataset import get_loaders
from src.model import MultiTaskCNN
from src.train import train_model
from src.evaluate import evaluate


device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, test_loader, train_ds = get_loaders("src/dataset")

model = MultiTaskCNN(
    num_plants=train_ds.num_plants,
    num_diseases=train_ds.num_diseases
)

train_model(model, train_loader, val_loader, epochs=15, device=device)
evaluate(model, test_loader, device)
