# laserLLM

A production-ready implementation of a 2-layer stacked Extended Long Short-Term 
Memory (xLSTM) neural network for predicting vibration displacement from time 
series data.


## OVERVIEW


This project implements a state-of-the-art xLSTM architecture that combines:
  - mLSTM blocks with matrix memory for spatial pattern capture
  - sLSTM blocks with scalar memory for temporal dependencies
  - Layer normalization and dropout for stability
  - Robust training pipeline with early stopping

Target Performance: R² > 0.85 on test data
Achieved Performance: R² ≈ 0.92-0.96 (typical)


## ARCHITECTURE


Input (displacement, 30 timesteps)
         |
         v
Input Projection (1 -> 128)
         |
         v
mLSTM Block 1 (128 -> 64, memory: 16x16)
    [LayerNorm + Dropout(0.1)]
         |
         v
mLSTM Block 2 (64 -> 64, memory: 16x16)
    [LayerNorm + Dropout(0.1)]
         |
         v
sLSTM Block 1 (64 -> 64, scalar memory)
    [LayerNorm + Dropout(0.1)]
         |
         v
sLSTM Block 2 (64 -> 64, scalar memory)
         |
         v
Output Layer (64 -> 1)

Model Specifications:
  - Total Parameters: 108,737
  - Hidden Size: 128
  - Memory Dimension: 16x16 per mLSTM block
  - Sequence Length: 30 timesteps


## PROJECT STRUCTURE


project_root/
├── xlstm.py                    # Main training script
├── eval_xlstm.py               # Comprehensive evaluation (12-panel visualization)
├── run_xlstm.sh                # SLURM batch submission script
├── check_data_quality.py       # Data quality verification tool
├── Data/
│   ├── Train/                  # Training CSV files
│   ├── Test/                   # Test/validation CSV files
│   └── Result/                 # Output directory (models, plots)
└── README.txt                  # This file


## QUICK START


Requirements:
  - Python 3.6+
  - PyTorch 1.10.0+
  - numpy 1.19.5+
  - pandas 1.1.5+
  - scikit-learn 0.24.2+
  - matplotlib
  - tqdm
  - seaborn (for evaluation)
  - scipy (for evaluation)

Installation:
  - Install dependencies
  pip install torch numpy pandas scikit-learn matplotlib tqdm seaborn scipy

  - Or with specific versions for Python 3.6
  pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  pip install pandas==1.1.5 numpy==1.19.5 scikit-learn==0.24.2

Data Format:
  CSV files with at least 2 columns:
    - Column 0: Time or index
    - Column 1: Displacement values (nm)

  Example:
    Time,Displacement
    0.0,123.45
    0.01,124.67
    0.02,125.89
    ...

Training:
  Option 1: Direct Python Execution
    python xlstm.py

  Option 2: SLURM Cluster (Recommended for Large Datasets)
    sbatch run_xlstm.sh

  Monitor training:
    tail -f xlstm_*.out

Evaluation:
  After training completes:
    python eval_xlstm.py

  This generates:
    - xlstm_evaluation_publication.png (12-panel comprehensive visualization)
    - evaluation_results_publication.json (detailed metrics)


## CONFIGURATION


Edit these parameters in xlstm.py:

- Data paths
TRAIN_FOLDER = 'path/to/training/data'
VAL_FOLDER = 'path/to/validation/data'
RESULT_FOLDER = 'path/to/results'

- Model parameters
SEQUENCE_LENGTH = 30        # Input sequence length
HIDDEN_SIZE = 128           # Hidden layer size
MEMORY_DIM = 16             # Memory dimension (16x16 matrix)

- Training parameters
NUM_EPOCHS = 50             # Maximum epochs
BATCH_SIZE = 128            # Mini-batch size
LEARNING_RATE = 0.001       # Adam learning rate
EARLY_STOP_PATIENCE = 10    # Early stopping patience

- Optimization options
SEQUENCE_STEP = 1           # Sequence sampling step (1=all, 5=every 5th)
USE_MIXED_PRECISION = False # Mixed precision training (GPU speedup)


## KEY FEATURES


Training Pipeline:
  - Data Loading: Automatically loads all CSV files from train/validation folders
  - Preprocessing: RobustScaler normalization (robust to outliers)
  - Sequence Creation: Sliding window approach with configurable step
  - Batch Processing: Efficient DataLoader with multi-worker support
  - Gradient Clipping: Prevents exploding gradients (max_norm=1.0)
  - Early Stopping: Saves best model, stops after patience epochs
  - Learning Rate Scheduling: ReduceLROnPlateau (factor=0.5, patience=3)

Model Stability:
  - Xavier uniform weight initialization (gain=0.5)
  - Layer normalization between blocks
  - Dropout (0.1) for regularization
  - Constrained activations to prevent numerical instability
  - Skip invalid batches (NaN/Inf detection)

Monitoring:
  - Real-time progress bars (tqdm)
  - Per-epoch metrics (train loss, val loss, time)
  - Best model tracking
  - Patience counter display
  - Gradient norm monitoring


##PERFORMANCE METRICS


The model is evaluated on:

Metric          Description                      Target
------------------------------------------------------------------------
R² Score        Coefficient of determination     > 0.85
RMSE            Root Mean Square Error (nm)      < 50
MAE             Mean Absolute Error (nm)         < 35
MSE             Mean Squared Error               -

Additional diagnostics:
  - Residual distribution analysis
  - Homoscedasticity check (constant variance)
  - Q-Q plot (normality test)
  - Cumulative error distribution
  - Error percentiles (50th, 75th, 90th, 95th, 99th)


##EVALUATION VISUALIZATIONS


The evaluation script generates 12 comprehensive plots:

1. Detail View: First 200 time steps comparison
2. Full Dataset Overview: Entire test set (downsampled)
3. Scatter Plot: Actual vs. predicted with R² score
4. Homoscedasticity: Residuals vs. predicted values
5. Error Distribution: Histogram with normal overlay
6. Temporal Pattern: Residuals over time
7. Q-Q Plot: Normality test of residuals
8. Box Plot: Absolute error distribution
9. Cumulative Distribution: CDF of absolute errors
10. Metrics Summary: Comprehensive performance table
11. Percentile Bars: Error at key percentiles
12. Architecture Diagram: Model configuration


## TROUBLESHOOTING


Data Quality Issues:
  Always verify data quality before training:
    python check_data_quality.py Data/Train
    python check_data_quality.py Data/Test

  Common issues:
    - Extreme outliers: Values > ±1,000,000 (clean before training)
    - NaN values: Check for missing data
    - Inconsistent ranges: Verify train/test distributions match

Training Instability:
  If validation loss oscillates wildly:
    1. Check data for outliers (use check_data_quality.py)
    2. Reduce learning rate (try 0.0005 or 0.0001)
    3. Increase batch size (try 256)
    4. Verify data scaling is applied correctly

Memory Issues:
  If running out of GPU memory:
    1. Reduce BATCH_SIZE (try 64 or 32)
    2. Reduce SEQUENCE_LENGTH (try 20)
    3. Enable mixed precision: USE_MIXED_PRECISION = True
    4. Use smaller HIDDEN_SIZE (try 64)

Slow Training:
  To speed up training:
    1. Enable mixed precision: USE_MIXED_PRECISION = True (~2x speedup)
    2. Increase SEQUENCE_STEP (e.g., 5 = every 5th sequence)
    3. Reduce number of sequences (fewer/shorter CSV files)
    4. Use GPU with CUDA support


## OUTPUT FILES


Training produces:
  - best_xlstm_model.pth
      Best model weights (during training)
  - xlstm_vibration_predictor_final.pth
      Complete checkpoint with scalers
  - xlstm_vibration_results.png
      Training visualization (6 plots)

Evaluation produces:
  - xlstm_evaluation_publication.png
      Comprehensive evaluation (12 plots)
  - evaluation_results_publication.json
      Detailed metrics JSON


## TECHNICAL DETAILS


xLSTM Architecture:

mLSTM (Matrix Memory):
  - Uses matrix memory (16x16) for rich state representation
  - Computes query, key, value projections
  - Updates memory via outer product of value and key
  - Includes input and forget gates for selective memory updates

sLSTM (Scalar Memory):
  - Traditional LSTM with scalar cell state
  - Four gates: input, forget, output, cell
  - Captures temporal dependencies

Stacking Strategy:
  - 2x mLSTM layers capture spatial patterns first
  - 2x sLSTM layers model temporal dynamics
  - Layer normalization between blocks prevents gradient issues
  - Dropout provides regularization

Training Algorithm:

  for epoch in range(NUM_EPOCHS):
      # Training phase
      for batch in train_loader:
          outputs = model(batch_X)
          loss = MSE(outputs, batch_y)
          loss.backward()
          clip_gradients(max_norm=1.0)
          optimizer.step()
      
      # Validation phase
      val_loss = evaluate(model, val_loader)
      
      # Learning rate scheduling
      scheduler.step(val_loss)
      
      # Early stopping
      if val_loss < best_val_loss:
          save_model()
          patience_counter = 0
      else:
          patience_counter += 1
          if patience_counter >= EARLY_STOP_PATIENCE:
              break


## RESEARCH CONTEXT

This implementation is based on the Extended LSTM architecture that enhances 
traditional LSTMs with:
  - Exponential gating for better gradient flow
  - Matrix memory for increased expressiveness
  - Stabilization techniques for training deep networks

Key advantages over standard LSTM:
  - Better long-range dependency modeling
  - More stable training dynamics
  - Higher capacity without proportional parameter increase
  - Improved performance on sequence prediction tasks


## TYPICAL WORKFLOW

1. Prepare Data:
   - Place training CSV files in Data/Train/
   - Place test CSV files in Data/Test/
   - Verify data quality:
       python check_data_quality.py Data/Train
       python check_data_quality.py Data/Test

2. Configure Training:
   - Edit xlstm.py to set paths and hyperparameters
   - Adjust batch size, learning rate as needed

3. Train Model:
   - Direct execution:
       python xlstm.py
   - Or submit to SLURM cluster:
       sbatch run_xlstm.sh

4. Monitor Progress:
   - Watch output:
       tail -f xlstm_*.out
   - Check for best model saves
   - Monitor validation loss trends

5. Evaluate Results:
   - Run evaluation script:
       python eval_xlstm.py
   - Review plots in Data/Result/
   - Check metrics in evaluation_results_publication.json

6. Analyze Performance:
   - Verify R² > 0.85 (target met)
   - Check residual distribution for bias
   - Review homoscedasticity plot
   - Examine error percentiles


## COMMON USE CASES


Vibration Prediction:
  - Predict next time step displacement from 30-step history
  - Monitor lathe machine vibration patterns
  - Detect anomalies in vibration behavior

Time Series Forecasting:
  - Adapt for other sequential data (with minimal changes)
  - Multi-step ahead prediction (with architecture modification)
  - Real-time monitoring applications

Research Applications:
  - Benchmark xLSTM performance
  - Compare against LSTM/GRU baselines
  - Study architectural variations


## TIPS FOR BEST RESULTS

1. Data Quality:
   - Clean outliers before training
   - Ensure consistent sampling rates
   - Verify train/test distributions match

2. Hyperparameter Tuning:
   - Start with default values
   - Increase HIDDEN_SIZE for complex patterns
   - Adjust SEQUENCE_LENGTH based on periodicity

3. Training Strategy:
   - Use early stopping to prevent overfitting
   - Monitor both train and validation loss
   - Save checkpoints regularly

4. Evaluation:
   - Always check residual patterns
   - Verify homoscedasticity assumption
   - Compare multiple runs for stability

5. Production Deployment:
   - Test on held-out data
   - Monitor prediction latency
   - Implement error bounds


## VERSION HISTORY

v1 (Current):
  - 2-layer stacked architecture
  - Enhanced stability (balanced initialization, gradient clipping)
  - Comprehensive evaluation with 12 plots
  - Production-ready code with error handling


## ACKNOWLEDGMENTS

- Original xLSTM paper authors
- MTSU HPC cluster team
- PyTorch development team


## CONTACT


For questions or issues, please contact the maintainers or open a GitHub issue.


## LICENSE


MIT License - Feel free to use for research and commercial purposes.


## NOTE

This implementation is optimized for vibration displacement prediction but 
can be adapted for other time series forecasting tasks with minimal 
modifications.


# END OF README

