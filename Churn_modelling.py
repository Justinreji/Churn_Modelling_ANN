# Customer Churn Classification using Artificial Neural Networks

# ---------------------------------------------
# Step 1: Load and Explore the Dataset
# ---------------------------------------------
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("Churn_Modelling.csv")

# Drop irrelevant columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Display dataset structure
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nCheck for null values")
print(df.isnull().sum())


# ---------------------------------------------
# Step 2: Preprocessing and Feature Engineering
# ---------------------------------------------
X = df.drop(columns=["Exited"])
y = df["Exited"]

# Encode categorical features (Geography, Gender)
X_encoded = pd.get_dummies(X)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split dataset into training and validation sets with stratification
df_X = pd.DataFrame(X_scaled)
df_y = pd.Series(y.values)
X_train, X_valid, y_train, y_valid = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42, stratify=df_y
)

# ---------------------------------------------
# Step 3: Define PyTorch Dataset Class
# ---------------------------------------------
class ChurnData(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.to_numpy(), dtype=torch.long)
        self.len = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len

# ---------------------------------------------
# Step 4: Simplified Neural Network Architecture
# ---------------------------------------------
class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNet, self).__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.in_to_h1 = nn.Linear(input_dim, 64)
        self.h1_to_h2 = nn.Linear(64, 32)
        self.h2_to_out = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)  # regularization

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.in_to_h1(x))
        x = self.dropout(F.relu(self.h1_to_h2(x)))
        return self.h2_to_out(x)

# ---------------------------------------------
# Step 5: Training Function with Enhanced Evaluation
# ---------------------------------------------
def trainNN(epochs=100, batch_size=32, lr=0.001, decision_threshold=0.4):
    train_set = ChurnData(X_train, y_train)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    gn = ChurnNet(input_dim=X_train.shape[1])
    class_counts = y_train.value_counts().sort_index().values
    # Adjust class weights manually to emphasize the minority class
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(gn.parameters(), lr=lr)

    for epoch in range(epochs):
        gn.train()
        running_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            output = gn(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        gn.eval()
        with torch.no_grad():
            val_input = torch.tensor(X_valid.values, dtype=torch.float32)
            val_target = torch.tensor(y_valid.to_numpy(), dtype=torch.long)
            val_output = gn(val_input)
            probabilities = torch.softmax(val_output, dim=1)[:, 1]
            preds = (probabilities > decision_threshold).long()
            acc = accuracy_score(val_target, preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss / len(loader):.4f} | Accuracy: {acc:.4f}")

    # Final evaluation
    cm = confusion_matrix(val_target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Exited"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Customer Churn")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(val_target, preds, target_names=["Stayed", "Exited"]))

    return gn

# ---------------------------------------------
# Step 6: Train the Model
# ---------------------------------------------
trainNN()
