import torch
from torch import nn
import pandas as pd
import numpy as np

import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out


# Leggi i dati dai file CSV
pv_data = pd.read_csv('./dataset.csv', usecols=[1]).squeeze()
load_data = pd.read_csv('./dataset.csv', usecols=[2]).squeeze()
price_data = pd.read_csv('./dataset.csv', usecols=[3]).squeeze()

df = pd.DataFrame({
    'PV Data': pv_data.values,
    'Load Data': load_data.values,
    'Price Data': price_data.values
})


price = df['Price Data']

# Converto da euro/MWh a euro/kWh prima di usarlo nella funzione obiettivo
price = price / 1000

# Definizione della funzione per creare le finestre
def create_windows(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)

# Creazione delle finestre di dati
lookback = 168
forecast_horizon = 24

# Creo le feature da dare alla rete neurale per la previsione (prezzi, + eventuali variabili esogene)
input_features, _ = create_windows(price.values, lookback, forecast_horizon)

# Usiamo solo i dati di prezzo per i target
_, target_price = create_windows(price.values, lookback, forecast_horizon)


# Usiamo solo i dati di prezzo per i target
_, target_price = create_windows(price.values, lookback, forecast_horizon)

# Stampiamo le dimensioni per confermare che tutto sia corretto
print("Dimensioni delle features:", input_features.shape)        # Dovrebbe essere (num_windows, lookback*num_features)
print("Dimensioni delle target_price:", target_price.shape)  # Dovrebbe essere (num_windows, forecast_horizon)


from sklearn.model_selection import train_test_split


# Specifica la proporzione del set di test; ad esempio, 0.2 per il 20% come test set
test_size = 0.2

# Divide i dati in training e test set senza shuffle
input_features_train, input_features_test, target_price_train, target_price_test = train_test_split(
    input_features, 
    target_price, 
    test_size=test_size, 
    shuffle=False 
)



input_features_train = input_features_train.reshape(input_features_train.shape[0], lookback, -1)
input_features_test = input_features_test.reshape(input_features_test.shape[0], lookback, -1)


# Parametri del modello
input_dim = 1  # Dimensione delle features
hidden_dim = 48  # Dimensione dello stato nascosto
output_dim = target_price.shape[-1]  # Numero di step di previsione
num_layers = 1  # Numero di layer LSTM

# Inizializzazione del modello
reg = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

import torch.optim as optim

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(reg.parameters(), lr=0.001)  # Adam optimizer con un learning rate di 0.001

# Configurazione del dispositivo di calcolo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg.to(device)

# Conversione dei dati in tensori PyTorch e trasferimento al dispositivo appropriato
input_features_train = torch.tensor(input_features_train, dtype=torch.float32).to(device)
target_price_train = torch.tensor(target_price_train, dtype=torch.float32).to(device)
input_features_test = torch.tensor(input_features_test, dtype=torch.float32).to(device)
target_price_test = torch.tensor(target_price_test, dtype=torch.float32).to(device)

# Numero di epoche di training
num_epochs = 50
from torch.utils.data import TensorDataset, DataLoader

# Definizione del batch size
batch_size = 32

# Creazione dei TensorDataset e DataLoader per il training e il test
train_data = TensorDataset(input_features_train, target_price_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

test_data = TensorDataset(input_features_test, target_price_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Training del modello con il DataLoader
for epoch in range(num_epochs):
    reg.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = reg(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Test del modello con il DataLoader
reg.eval()
total_test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = reg(inputs)
        loss = criterion(outputs, targets)
        total_test_loss += loss.item()

print(f'Test Loss: {total_test_loss/len(test_loader):.4f}')


import matplotlib.pyplot as plt

# Assicurati di calcolare le previsioni solo sull'ultima finestra di test se non già fatto
reg.eval()
with torch.no_grad():
    last_inputs = input_features_test[-1].unsqueeze(0)  # Prendi l'ultima finestra di input e aggiungi una dimensione batch
    last_inputs = last_inputs.to(device)
    predicted_last = reg(last_inputs).cpu().numpy().flatten()  # Genera previsioni e sposta i dati sul CPU per il plotting

# Dati reali corrispondenti all'ultima finestra di test
actual_last = target_price_test[-1].cpu().numpy().flatten()

# Impostazione delle etichette di tempo per l'asse x
time_steps = range(len(predicted_last))

# Creazione del plot
plt.figure(figsize=(10, 6))
plt.plot(time_steps, actual_last, label='Actual Price')
plt.plot(time_steps, predicted_last, label='Predicted Price', linestyle='--')
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price (euro/kWh)')
plt.legend()
plt.grid(True)
plt.show()