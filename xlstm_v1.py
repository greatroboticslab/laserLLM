# Import Libraries
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import math
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Data paths
TRAIN_FOLDER = '/home/tc6d/ondemand/data/sys/myjobs/projects/default/1/Data/Train'
VAL_FOLDER = '/home/tc6d/ondemand/data/sys/myjobs/projects/default/1/Data/Test'
RESULT_FOLDER = '/home/tc6d/ondemand/data/sys/myjobs/projects/default/1/Data/Result'

# Model parameters
SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 128
MEMORY_DIM = 16

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 10

# Optimization options
SEQUENCE_STEP = 1  # Set to 5 or 10 to reduce sequences and speed up training
USE_MIXED_PRECISION = False  # Set to True for ~2x speedup on modern GPUs

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_vibration_data_from_folder(folder_path, folder_name="data"):
    """Load all vibration time series data from specified folder"""
    all_series = []
    file_names = []
    
    print(f"\n{'='*70}")
    print(f"Loading data from {folder_name} folder")
    print(f"Path: {folder_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder not found: {folder_path}")
        return all_series, file_names
    
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if len(csv_files) == 0:
        print(f"⚠️ WARNING: No CSV files found in {folder_path}")
        return all_series, file_names
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    total_points = 0
    for idx, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            if len(df.columns) >= 2:
                displacement_values = df.iloc[:, 1].values
                all_series.append(displacement_values)
                file_names.append(os.path.basename(file))
                total_points += len(displacement_values)
                print(f"[{idx:2d}/{len(csv_files)}] ✓ {os.path.basename(file):40s} - {len(displacement_values):,} points")
        except Exception as e:
            print(f"[{idx:2d}/{len(csv_files)}] ✗ Error loading {os.path.basename(file)}: {e}")
    
    print(f"\n{'='*70}")
    print(f"{folder_name} Data Summary:")
    print(f"  Files loaded: {len(all_series)}")
    print(f"  Total data points: {total_points:,}")
    print(f"  Average points per file: {total_points//len(all_series):,}" if all_series else "  N/A")
    print(f"{'='*70}\n")
    
    return all_series, file_names

def create_sequences_for_prediction(data, sequence_length=30, step=1):
    """Create sequences for time series prediction with optional step for speed"""
    X, y = [], []
    for signal in data:
        if len(signal) > sequence_length:
            for i in range(0, len(signal) - sequence_length, step):
                X.append(signal[i:(i + sequence_length)])
                y.append(signal[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_datasets(train_folder, val_folder, sequence_length=30, step=1):
    """Prepare datasets from separate train and validation folders"""
    
    start_time = time.time()
    
    # Load training data
    train_series, train_files = load_vibration_data_from_folder(train_folder, "Training")
    
    # Load validation data
    val_series, val_files = load_vibration_data_from_folder(val_folder, "Validation")
    
    if len(train_series) == 0:
        raise ValueError("❌ ERROR: No training data loaded!")
    if len(val_series) == 0:
        raise ValueError("❌ ERROR: No validation data loaded!")
    
    # Create sequences for training
    print(f"Creating sequences with length={sequence_length}, step={step}...")
    print(f"(Step={step} means using every {step}th sequence for speed)\n")
    
    X_train_list, y_train_list = [], []
    for idx, ts_data in enumerate(train_series, 1):
        X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)
        if len(X_ts) > 0:
            X_train_list.extend(X_ts)
            y_train_list.extend(y_ts)
            print(f"  Train file {idx}/{len(train_series)}: {len(X_ts):,} sequences created")
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    print(f"\nTotal training sequences: {len(X_train):,}\n")
    
    # Create sequences for validation
    X_val_list, y_val_list = [], []
    for idx, ts_data in enumerate(val_series, 1):
        X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)
        if len(X_ts) > 0:
            X_val_list.extend(X_ts)
            y_val_list.extend(y_ts)
            print(f"  Val file {idx}/{len(val_series)}: {len(X_ts):,} sequences created")
    
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Dataset Preparation Complete (took {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"Training sequences:   {X_train.shape[0]:,}")
    print(f"Validation sequences: {X_val.shape[0]:,}")
    print(f"Sequence length:      {sequence_length}")
    print(f"{'='*70}\n")
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    return X_train, X_val, y_train, y_val, train_series, val_series

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedmLSTMBlock(nn.Module):
    """Improved mLSTM block with balanced stability"""
    def __init__(self, input_size, hidden_size, mem_dim=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        
        # Linear transformations
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, mem_dim)
        self.Wv = nn.Linear(input_size, mem_dim)
        
        # Gates
        self.Wi = nn.Linear(input_size + hidden_size, mem_dim)
        self.Wf = nn.Linear(input_size + hidden_size, mem_dim)
        
        # Output gate - FIXED: correct input size
        self.Wo = nn.Linear(input_size + 1, hidden_size)
        
        # Stability layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Balanced weight initialization"""
        for layer in [self.Wq, self.Wk, self.Wv, self.Wi, self.Wf, self.Wo]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x, states):
        h_prev, m_prev = states
        
        # Moderate input constraints
        x_norm = torch.tanh(x) * 0.3
        
        # Query, Key, Value with balanced scaling
        qt = torch.tanh(self.Wq(x_norm)) * 0.5
        kt = torch.tanh(self.Wk(x_norm)) * (1.0 / math.sqrt(self.mem_dim))
        vt = torch.tanh(self.Wv(x_norm)) * 0.5
        
        # Gates with reasonable constraints
        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)
        it = torch.sigmoid(self.Wi(combined)) * 0.9 + 0.05
        ft = torch.sigmoid(self.Wf(combined)) * 0.9 + 0.05
        
        # Memory update with controlled scaling
        vt_kt_outer = torch.bmm(vt.unsqueeze(2), kt.unsqueeze(1)) * 0.1
        
        # Expand gates
        it_expanded = it.unsqueeze(2).expand_as(vt_kt_outer)
        ft_expanded = ft.unsqueeze(2).expand_as(vt_kt_outer)
        
        # Memory update
        m_t = ft_expanded * m_prev + it_expanded * vt_kt_outer
        
        # Output gate - FIXED: proper concatenation
        m_t_mean = m_t.mean(dim=(1, 2), keepdim=False)
        combined_output = torch.cat([x_norm, m_t_mean.unsqueeze(1)], dim=1)
        ot = torch.sigmoid(self.Wo(combined_output)) * 0.9 + 0.05
        
        # Hidden state with layer norm
        h_t = ot * torch.tanh(self.layer_norm(qt))
        h_t = self.dropout(h_t)
        
        return h_t, (h_t, m_t)

class ImprovedsLSTMBlock(nn.Module):
    """Improved sLSTM block with balanced stability"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Balanced weight initialization"""
        for layer in [self.W_i, self.W_f, self.W_o, self.W_c]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x, states):
        h_prev, c_prev = states
        
        # Moderate input constraints
        x_norm = torch.tanh(x) * 0.3
        
        # Combine
        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)
        
        # Gates with reasonable constraints
        i_t = torch.sigmoid(self.W_i(combined)) * 0.9 + 0.05
        f_t = torch.sigmoid(self.W_f(combined)) * 0.9 + 0.05
        o_t = torch.sigmoid(self.W_o(combined)) * 0.9 + 0.05
        c_hat_t = torch.tanh(self.W_c(combined)) * 0.5
        
        # Cell state update
        c_t = f_t * c_prev + i_t * c_hat_t
        
        # Hidden state with layer norm
        h_t = o_t * torch.tanh(self.layer_norm(c_t))
        h_t = self.dropout(h_t)
        
        return h_t, (h_t, c_t)

class ImprovedxLSTMPredictor(nn.Module):
    """Improved xLSTM with 2-layer stacked blocks for better performance"""
    def __init__(self, input_size=1, hidden_size=128, mem_dim=16, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # STACKED xLSTM components (2 layers each)
        # Layer 1: mLSTM blocks
        self.mlstm1 = ImprovedmLSTMBlock(hidden_size, hidden_size//2, mem_dim=mem_dim)
        self.mlstm2 = ImprovedmLSTMBlock(hidden_size//2, hidden_size//2, mem_dim=mem_dim)
        
        # Layer 2: sLSTM blocks
        self.slstm1 = ImprovedsLSTMBlock(hidden_size//2, hidden_size//2)
        self.slstm2 = ImprovedsLSTMBlock(hidden_size//2, hidden_size//2)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size//2, output_size)
        
        # Stability layers (between stacked blocks)
        self.layer_norm1 = nn.LayerNorm(hidden_size//2)
        self.layer_norm2 = nn.LayerNorm(hidden_size//2)
        self.layer_norm3 = nn.LayerNorm(hidden_size//2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Balanced initialization"""
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for all stacked layers"""
        # mLSTM layer 1
        h_mlstm1 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        m_mlstm1 = torch.zeros(batch_size, self.mem_dim, self.mem_dim, device=device)
        
        # mLSTM layer 2
        h_mlstm2 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        m_mlstm2 = torch.zeros(batch_size, self.mem_dim, self.mem_dim, device=device)
        
        # sLSTM layer 1
        h_slstm1 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        c_slstm1 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        
        # sLSTM layer 2
        h_slstm2 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        c_slstm2 = torch.zeros(batch_size, self.hidden_size//2, device=device)
        
        return [
            (h_mlstm1, m_mlstm1),
            (h_mlstm2, m_mlstm2),
            (h_slstm1, c_slstm1),
            (h_slstm2, c_slstm2)
        ]
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        if not hasattr(self, 'device'):
            self.device = x.device
        
        hidden_states = self.init_hidden(batch_size, self.device)
        
        # Process sequence through stacked layers
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Input projection
            x_proj = torch.tanh(self.input_proj(x_t)) * 0.5
            
            # mLSTM Layer 1
            x_proj, hidden_states[0] = self.mlstm1(x_proj, hidden_states[0])
            x_proj = self.layer_norm1(x_proj)
            x_proj = self.dropout1(x_proj)
            
            # mLSTM Layer 2
            x_proj, hidden_states[1] = self.mlstm2(x_proj, hidden_states[1])
            x_proj = self.layer_norm2(x_proj)
            x_proj = self.dropout2(x_proj)
            
            # sLSTM Layer 1
            x_proj, hidden_states[2] = self.slstm1(x_proj, hidden_states[2])
            x_proj = self.layer_norm3(x_proj)
            x_proj = self.dropout3(x_proj)
            
            # sLSTM Layer 2
            x_proj, hidden_states[3] = self.slstm2(x_proj, hidden_states[3])
        
        # Output
        output = self.output_layer(x_proj)
        
        return output

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def validate(model, val_loader, criterion, device, scaler=None):
    """Validation loop"""
    model.eval()
    val_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Skip invalid batches
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                continue
            
            if scaler and USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
            else:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                val_loss += loss.item()
                valid_batches += 1
    
    return val_loss / max(valid_batches, 1)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs, scaler=None):
    """Training loop for one epoch"""
    model.train()
    epoch_loss = 0
    valid_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_X, batch_y in progress_bar:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Skip invalid batches
        if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
            continue
        if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
            continue
        
        optimizer.zero_grad()
        
        if scaler and USE_MIXED_PRECISION:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 10.0:
                optimizer.zero_grad()
                continue
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 10.0:
                optimizer.zero_grad()
                continue
            
            optimizer.step()
        
        epoch_loss += loss.item()
        valid_batches += 1
        
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Grad': f'{total_norm:.4f}'
        })
    
    return epoch_loss / max(valid_batches, 1), valid_batches

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*70)
print(" xLSTM VIBRATION PREDICTION TRAINING")
print(" 2-Layer Stacked Architecture")
print("="*70)
print(f"\nConfiguration:")
print(f"  Sequence Length: {SEQUENCE_LENGTH}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Memory Dim: {MEMORY_DIM}")
print(f"  Stacking: 2 mLSTM layers + 2 sLSTM layers")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Epochs: {NUM_EPOCHS}")
print(f"  Early Stop Patience: {EARLY_STOP_PATIENCE}")
print(f"  Sequence Step: {SEQUENCE_STEP}")
print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
print("="*70)

# Prepare datasets
X_train, X_val, y_train, y_val, train_series, val_series = prepare_datasets(
    train_folder=TRAIN_FOLDER,
    val_folder=VAL_FOLDER,
    sequence_length=SEQUENCE_LENGTH,
    step=SEQUENCE_STEP
)

print(f"Data ranges:")
print(f"  Training y:   [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
print(f"  Validation y: [{np.min(y_val):.2f}, {np.max(y_val):.2f}]")

# Scale data
print(f"\nScaling data...")
feature_scaler = RobustScaler()
target_scaler = RobustScaler()

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

X_train_scaled = feature_scaler.fit_transform(X_train_flat)
X_val_scaled = feature_scaler.transform(X_val_flat)

y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_val_scaled = X_val_scaled.reshape(X_val.shape)

print(f"✅ Data normalized!")

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

print(f"✅ Tensors created!")

# Initialize model
model = ImprovedxLSTMPredictor(
    input_size=1,
    hidden_size=HIDDEN_SIZE,
    mem_dim=MEMORY_DIM,
    output_size=1
).to(device)

print(f"\n✅ xLSTM model initialized:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Architecture: STACKED xLSTM (2-layer)")
print(f"    Input → {HIDDEN_SIZE} → mLSTM1 ({HIDDEN_SIZE//2}) → mLSTM2 ({HIDDEN_SIZE//2})")
print(f"         → sLSTM1 ({HIDDEN_SIZE//2}) → sLSTM2 ({HIDDEN_SIZE//2}) → Output (1)")
print(f"  Memory Dimension: {MEMORY_DIM}x{MEMORY_DIM} per mLSTM block")

# Test forward pass
print(f"\nTesting forward pass...")
test_batch = X_train_tensor[:32].to(device)
with torch.no_grad():
    test_output = model(test_batch)
    print(f"✅ Output shape: {test_output.shape}")
    print(f"✅ Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

# Data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"\n{'='*70}")
print(f" STARTING TRAINING")
print(f"{'='*70}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"{'='*70}\n")

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
training_start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    
    # Train
    train_loss, valid_batches = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS, scaler
    )
    
    # Validate
    val_loss = validate(model, val_loader, criterion, device, scaler)
    
    epoch_time = time.time() - epoch_start_time
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"\n{'='*70}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {epoch_time:.1f}s")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model_path = os.path.join(RESULT_FOLDER, 'best_xlstm_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"  ✅ New best model saved!")
    else:
        patience_counter += 1
        print(f"  ⏳ Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
    
    # Estimate remaining time
    if epoch == 0:
        estimated_total_time = epoch_time * NUM_EPOCHS
        print(f"  📊 Estimated total time: {estimated_total_time/3600:.1f} hours")
    
    print(f"{'='*70}")
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
        break

training_time = time.time() - training_start_time
print(f"\n{'='*70}")
print(f" TRAINING COMPLETED")
print(f"{'='*70}")
print(f"Total training time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"{'='*70}\n")

# Load best model
model_path = os.path.join(RESULT_FOLDER, 'best_xlstm_model.pth')
model.load_state_dict(torch.load(model_path))
print("✅ Loaded best model for evaluation\n")

# EVALUATION
print("="*70)
print(" EVALUATING MODEL")
print("="*70)

model.eval()
predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(X_val_tensor), BATCH_SIZE), desc="Evaluating"):
        batch_X = X_val_tensor[i:i+BATCH_SIZE].to(device)
        
        if scaler and USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
        else:
            outputs = model(batch_X)
        
        predictions.append(outputs.cpu().numpy())

y_pred_scaled = np.concatenate(predictions).flatten()

# Inverse transform
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

# Remove invalid values
valid_mask = np.isfinite(y_actual) & np.isfinite(y_pred)
y_actual = y_actual[valid_mask]
y_pred = y_pred[valid_mask]

# Calculate metrics
mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)
residuals = y_actual - y_pred

print(f"\n{'='*70}")
print(f" RESULTS")
print(f"{'='*70}")
print(f"\nPerformance Metrics:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f} nm")
print(f"  MAE:  {mae:.4f} nm")
print(f"  R²:   {r2:.4f}")
print(f"\nDataset Info:")
print(f"  Training sequences:   {len(X_train_tensor):,}")
print(f"  Validation sequences: {len(y_actual):,}")
print(f"  Best validation loss: {best_val_loss:.6f}")
print(f"\nTraining Info:")
print(f"  Total time: {training_time/3600:.2f} hours")
print(f"  Epochs completed: {len(train_losses)}")
print(f"\nResiduals:")
print(f"  Mean: {np.mean(residuals):.4f}")
print(f"  Std:  {np.std(residuals):.4f}")
print(f"  Min:  {np.min(residuals):.4f}")
print(f"  Max:  {np.max(residuals):.4f}")
print(f"{'='*70}\n")

# Plots
fig = plt.figure(figsize=(16, 10))

# Plot 1: Training history
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_losses, label='Training Loss', linewidth=2)
ax1.plot(val_losses, label='Validation Loss', linewidth=2)
ax1.set_title('xLSTM Training History', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual
ax2 = plt.subplot(2, 3, 2)
plot_points = min(300, len(y_actual))
ax2.plot(y_actual[:plot_points], label='Actual', alpha=0.8, linewidth=2)
ax2.plot(y_pred[:plot_points], label='Predicted', alpha=0.8, linewidth=1.5)
ax2.set_title(f'Predictions vs Actual (First {plot_points} Points)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Displacement (nm)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter plot
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_actual, y_pred, alpha=0.4, s=2)
ax3.plot([y_actual.min(), y_actual.max()], 
         [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect')
ax3.set_xlabel('Actual Displacement (nm)')
ax3.set_ylabel('Predicted Displacement (nm)')
ax3.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals histogram
ax4 = plt.subplot(2, 3, 4)
ax4.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.set_title('Prediction Errors', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals over time
ax5 = plt.subplot(2, 3, 5)
ax5.plot(residuals[:plot_points], alpha=0.6, linewidth=1)
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Sample Index')
ax5.set_ylabel('Residual')
ax5.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
metrics_text = f"""
PERFORMANCE SUMMARY

MSE:  {mse:.4f}
RMSE: {rmse:.4f} nm
MAE:  {mae:.4f} nm
R²:   {r2:.4f}

Training Time: {training_time/3600:.2f}h
Best Val Loss: {best_val_loss:.6f}
Epochs: {len(train_losses)}

Train Samples: {len(X_train_tensor):,}
Val Samples:   {len(y_actual):,}

Model: xLSTM
Hidden: {HIDDEN_SIZE}
Memory: {MEMORY_DIM}
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
         verticalalignment='center')
ax6.set_title('2-Layer Stacked xLSTM', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(RESULT_FOLDER, 'xlstm_vibration_results.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Results plot saved: {plot_path}")
plt.show()

# Save model
save_dict = {
    'model_state_dict': model.state_dict(),
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler,
    'sequence_length': SEQUENCE_LENGTH,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss,
    'training_time_hours': training_time / 3600,
    'metrics': {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    },
    'config': {
        'hidden_size': HIDDEN_SIZE,
        'memory_dim': MEMORY_DIM,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'sequence_step': SEQUENCE_STEP,
        'mixed_precision': USE_MIXED_PRECISION
    }
}

torch.save(save_dict, os.path.join(RESULT_FOLDER, 'xlstm_vibration_predictor_final.pth'))
print(f"✅ Model saved: {os.path.join(RESULT_FOLDER, 'xlstm_vibration_predictor_final.pth')}")

print("\n" + "="*70)
print(" TRAINING AND EVALUATION COMPLETE!")
print("="*70)
print(f"\n📁 All results saved to: {RESULT_FOLDER}")
print(f"   - best_xlstm_model.pth (best model during training)")
print(f"   - xlstm_vibration_results.png (visualization)")
print(f"   - xlstm_vibration_predictor_final.pth (final model + all data)")
print("="*70)