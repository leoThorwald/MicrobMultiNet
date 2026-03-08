import torch.nn as nn
import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    

def build_mlp(input_dim, hidden_layers, hidden_width, output_dim, activation="gelu"):
    """
    Returns a new MLP model every time you call it.

    input_dim       : input feature size
    hidden_layers   : number of hidden layers
    hidden_width    : neurons per hidden layer
    output_dim      : output size
    activation      : activation function name
    """

    layers = []
    act = get_activation(activation)

    # first layer
    layers.append(nn.Linear(input_dim, hidden_width))
    layers.append(act)

    # hidden layers
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_width, hidden_width))
        layers.append(get_activation(activation))

    # output layer
    layers.append(nn.Linear(hidden_width, output_dim))

    return nn.Sequential(*layers)



def compute_metrics_with_threshold(C, threshold=0.01):
    """
    Compute accuracy, precision, and recall for predicted matrix C against ground truth B.
    Both matrices must have the same shape.
    Diagonal elements are ignored.
    C values above threshold are treated as 1, below as 0.
    """
    glv = np.load("/home/leo/Documents/MDSINE/NeuralODE/data/glv_params.npy")
    B = (glv != 0).astype(int)

# Set diagonal to 0
    np.fill_diagonal(B, 0)
    if B.shape != C.shape:
        raise ValueError("Matrices must have the same shape")
    
    # Ignore diagonal
    mask = ~np.eye(B.shape[0], dtype=bool)
    
    B_masked = B[mask]
    C_masked = (np.abs(C) > threshold)[mask].astype(int)  # thresholding
    
    TP = np.sum((B_masked == 1) & (C_masked == 1))
    FP = np.sum((B_masked == 0) & (C_masked == 1))
    FN = np.sum((B_masked == 1) & (C_masked == 0))
    TN = np.sum((B_masked == 0) & (C_masked == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return accuracy, precision, recall



def plot_results(dataset, func, t, method="euler", rtol=1e-5, atol=1e-7, inx=72, title_prefix="Neural ODE", augment_dim=0):
    """
    Beautiful multi-variable trajectory plot comparing ground truth vs Neural ODE prediction.
    """
    with torch.no_grad():
        initial_state1 = dataset[0, :].unsqueeze(0)
        inital_state2 = dataset[inx, :].unsqueeze(0)

        initial_state1 = torch.nn.functional.pad(initial_state1, (0, augment_dim))

        initial_state2 = torch.nn.functional.pad(inital_state2, (0, augment_dim))

        pred_y1 = odeint(func, initial_state1, t[:inx], method=method, rtol=rtol, atol=atol)
        pred_y2 = odeint(func, initial_state2, t[inx:], method=method, rtol=rtol, atol=atol)
        pred_y = torch.cat((pred_y1, pred_y2), dim=0)
        pred_y = pred_y.squeeze(1)

    # Convert to numpy
    true_y = dataset.numpy()      # (T, D)
    pred_y = pred_y.numpy()       # (T, D)
    time = t.numpy().flatten()

    # Variable names – customize if you have real names!
    var_names = [f"ASV {i}" for i in range(dataset.shape[1])]

    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), dpi=120)
    axes = axes.flatten()

    # Colors – professional palette
    color_true = '#2E86AB'   # Deep blue
    color_pred = '#A23B72'   # Purple/wine

    for i in range(dataset.shape[1]):
        ax = axes[i]
        
        # Plot ground truth and prediction
        ax.plot(time, true_y[:, i], color=color_true, linewidth=2.2, label='Ground Truth', alpha=0.9)
        ax.plot(time, pred_y[:, i], color=color_pred, linewidth=2.4, label='Neural ODE Prediction', alpha=0.95)

        # Styling
        ax.set_title(var_names[i], fontsize=13, fontweight='medium', pad=10)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
        ax.set_facecolor('#f8f9fa')  # Very light gray background for contrast
        mse = np.mean((true_y[:, i] - pred_y[:, i])**2)
        ax.set_title(f'{var_names[i]}  (MSE = {mse:.4f})', fontsize=13)

        # X-axis formatting (especially nice if `t` contains dates or datetime)
        if np.issubdtype(time.dtype, np.datetime64) or hasattr(time[0], 'date'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time)//6)))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        # Legend only on first subplot to avoid clutter
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=False, loc='upper left')

        # Tight layout per subplot
        ax.margins(x=0.01)

    # Hide unused subplots if dataset has <12 variables
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    

    # Global title and layout
    fig.suptitle(f'{title_prefix} — {method.upper()} Integration (rtol={rtol}, atol={atol})',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
    plt.show()

    return fig, axes