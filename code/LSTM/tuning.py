import optuna
import optuna.visualization as vis
from train_LSTM import train_model
from utils import create_dataset
from models import LSTMModel
import torch
import torch.nn as nn
import numpy as np

def objective(trial, data, n_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define hyperparameters
    lookback = trial.suggest_int("lookback", 5, 60, step=5)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_nodes = trial.suggest_int("n_nodes", 10, 100, step=10)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    # Create dataset with the current trial's lookback value
    X_train, X_val, _, y_train, y_val, _ = create_dataset(data, lookback=lookback, window_size=50, val_step=10, test_step=30)

    # train then get validation loss
    model = LSTMModel(input_dim=data.shape[1], n_nodes=n_nodes, output_dim=1, n_layers=n_layers, dropout_rate=dropout_rate)
    #model.double()
    model.to(device)
    blocks = list(X_train.keys())
    num_blocks = len(blocks)
    val_loss_list = np.zeros(num_blocks) # list of validation loss for each block

    for i, block in enumerate(blocks): # train block by block
        # Move data to the selected device
        X_train_block = X_train[block].to(device)
        y_train_block = y_train[block].to(device)
        X_val_block = X_val[block].to(device)
        y_val_block = y_val[block].to(device)
        
        _, best_set = train_model(X_train_block, y_train_block, model, lr=lr, n_epochs=n_epochs)
        # evaluate with val set of single block
        model.load_state_dict(best_set)
        model.eval()
        with torch.no_grad():
            output = model(X_val_block)
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, y_val_block)
        val_loss_list[i] = loss.item()

    return val_loss_list.mean()

def study_early_stop(study, trial):
    # Stop if no improvement in the last N trials
    N = 100
    threshold = 0.001

    if len(study.trials) < N:
        return

    values = [t.value for t in study.trials[-N:]]
    best_value = min(values)
    if all((abs(v - best_value) < threshold) or (v > best_value) for v in values):
        study.stop()

def tune_model(data, n_trails=200, n_epochs=10, baseline=False):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, data, n_epochs), n_trials=n_trails, callbacks=[study_early_stop])

    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value

    # Plot optimization history
    history = vis.plot_optimization_history(study)
    history.show()
    if baseline:
        history.write_image('../../output/LSTM_results/optimization_history_baseline.png')
    else:
        history.write_image('../../output/LSTM_results/optimization_history_proposed.png')

    # Plot parameter relationship
    importance = vis.plot_param_importances(study)
    importance.show()
    if baseline:
        importance.write_image('../../output/LSTM_results/param_importance_baseline.png')
    else:
        importance.write_image('../../output/LSTM_results/param_importance_proposed.png')
    
    # Plot slice of the parameters
    slice = vis.plot_slice(study, params=['n_layers', 'n_nodes', 'dropout_rate', 'lr'])
    slice.show()
    if baseline:
        slice.write_image('../../output/LSTM_results/param_slice_baseline.png')
    else:
        slice.write_image('../../output/LSTM_results/param_slice_proposed.png')


    return best_params, best_val_loss
