import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# -----------------------------
# 1. Define your Dataset Class
# -----------------------------
class CustomSpectrogramDataset(Dataset):
    def __init__(self, root_dir, folds=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Convert a single fold to a list if needed
        if folds is None:
            self.folds = list(range(1, 11))
        elif isinstance(folds, int):
            self.folds = [folds]
        else:
            self.folds = folds
        
        # Create a list of all image file paths only for the specified folds
        self.image_paths = []
        for fold in self.folds:
            fold_dir = os.path.join(root_dir, f'fold{fold}')
            if not os.path.isdir(fold_dir):
                raise ValueError(f"Directory {fold_dir} does not exist.")
            for file in os.listdir(fold_dir):
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(fold_dir, file))
        
        self.image_paths.sort()
        
        # Define the 10 UrbanSound8K classes for reference
        self.classes = [
            'air_conditioner', 
            'car_horn', 
            'children_playing', 
            'dog_bark',
            'drilling', 
            'engine_idling', 
            'gun_shot', 
            'jackhammer',
            'siren', 
            'street_music'
        ]
        
    def _extract_label(self, file_path):
        """Extracts the numeric class label from the filename (e.g., '74364-8-1-0.png')."""
        filename = os.path.basename(file_path)
        label = int(filename.split('-')[1])  # Extracts '8' in '74364-8-1-0.png'
        return label

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image in grayscale
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        if self.transform:
            image = self.transform(image)
        
        label = self._extract_label(img_path)
        
        return image, label

# -----------------------------
# 2. Define your transforms
# -----------------------------
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -----------------------------
# 3. Define your Model
# -----------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # --- First conv block: three conv layers with pooling ---
        # After each conv + pool, the spatial size halves:
        # 128 -> 64 -> 32 -> 16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # output: 32 x 128 x 128
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # output: 64 x 64 x 64
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 128 x 32 x 32
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimensions
        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # First conv block:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> 32 x 64 x 64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> 64 x 32 x 32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> 128 x 16 x 16
        
        # Transition: flatten and pass through FC layer, then reshape to a feature map.
        x = x.view(x.size(0), -1)                     # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)    
        x = self.fc2(x) 
        
        return x

# -----------------------------
# 4. Main Training Function with Early Stopping
# -----------------------------
def main():
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.0007
    patience = 3  # Number of epochs to wait for improvement before stopping early

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # List to store accuracy for each fold
    fold_accuracies = []

    # Loop over 10 folds
    for fold in range(1, 11):
        # Use the current fold as validation and the rest for training
        train_folds = [f for f in range(1, 11) if f != fold]
        val_folds = [fold]
        
        # Instantiate datasets for training and validation
        train_dataset = CustomSpectrogramDataset(
            root_dir='C:/Users/Aaron/Desktop/Uni/ScyPy/spectrograms', 
            folds=train_folds, 
            transform=data_transforms
        )
        val_dataset = CustomSpectrogramDataset(
            root_dir='C:/Users/Aaron/Desktop/Uni/ScyPy/spectrograms', 
            folds=val_folds, 
            transform=data_transforms
        )
        
        # Create DataLoaders (use pin_memory=True for speed on GPU)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=6,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=6,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        # Reinitialize model and optimizer for each fold, and move model to device
        model = SimpleCNN(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"Starting fold {fold}...")
        
        best_val_acc = 0.0
        epochs_without_improve = 0
        
        # Training loop for the current fold
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Wrap the training loader with tqdm
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{num_epochs}", leave=False)
            for images, labels in pbar:
                # Move images and labels to device
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                
                # Compute training accuracy
                _, preds = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (preds == labels).sum().item()

                # Update progress bar description with current loss and accuracy
                pbar.set_postfix(loss=loss.item(), acc=100.0 * correct_train / total_train)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_accuracy = 100.0 * correct_train / total_train
            
            # Evaluation on the validation fold after each epoch
            model.eval()
            correct = 0
            total = 0
            val_pbar = tqdm(val_loader, desc=f"Fold {fold} Validation", leave=False)
            with torch.no_grad():
                for images, labels in val_pbar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    val_pbar.set_postfix(acc=100.0 * correct / total)
            
            val_accuracy = 100.0 * correct / total
            print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Early stopping: if validation accuracy improves, reset counter; otherwise increment.
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
            
            if epochs_without_improve >= patience:
                print(f"No improvement for {patience} epochs. Early stopping on fold {fold}.")
                break  # Exit the epoch loop early
        
        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold} Best Validation Accuracy: {best_val_acc:.2f}%\n")

    torch.save(model.state_dict(), "simple_cnn_kleiner+bn.pth")
    print("Model saved successfully!")   
        
    # Display overall cross-validation results
    print("10-Fold Cross Validation Results:")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"Fold {i}: {acc:.2f}%")
    print(f"Average Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.2f}%")

# -----------------------------
# 5. Entry Point for Windows
# -----------------------------
if __name__ == "__main__":
    # This ensures child processes are started correctly on Windows
    mp.set_start_method("spawn", force=True)
    
    main()
