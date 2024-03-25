import torch

from src.data_preprocessing import DataPreprocessor
from src.train import train_model
from src.valid import evaluate_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r'C:\Users\Naeem\Desktop\car_bike_classification\data'
    batch_size = 32
    image_size = 224
    num_epochs = 2
    data_processor = DataPreprocessor(data_dir, batch_size, image_size)
    data_loaders = data_processor.create_data_loaders()

    print("----------Training Start----------------")
    model = train_model(data_loaders, num_epochs, device)
    print("----------Training Completed----------------")
    evaluate_model(model, data_loaders['test'], device)
    torch.save(model.state_dict(), 'model.pth')
if __name__ == "__main__":
    main()