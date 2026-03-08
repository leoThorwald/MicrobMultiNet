from utils import *
import os
import pandas as pd



class MicrobMultiNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_width, activation="tanh"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_width = hidden_width


        self.act = get_activation(activation)



    # Input → hidden
        self.W1 = nn.Parameter(torch.empty(input_dim, hidden_width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(input_dim, hidden_width))

        for i in range(input_dim):
            nn.init.xavier_uniform_(self.W1[i], gain=0.001)

        # Hidden layers
        self.W_hidden = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, hidden_width, hidden_width))
            for _ in range(hidden_layers)
        ])
        self.b_hidden = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, hidden_width))
            for _ in range(hidden_layers)
        ])

        for layer in self.W_hidden:
            for i in range(input_dim):
                nn.init.xavier_uniform_(layer[i], gain=0.001)

        # Output layer: hidden → 1
        self.W_out = nn.Parameter(torch.empty(input_dim, 1, hidden_width))
        self.b_out = nn.Parameter(torch.zeros(input_dim, 1))

        for i in range(input_dim):
            nn.init.xavier_uniform_(self.W_out[i], gain=0.001)


        # ---- Learnable mask ----
        self.mask = nn.Parameter(torch.ones(input_dim, input_dim))  # [N, N]

    def forward(self, t, y):
        """
        y: tensor of shape [input_dim]
        returns: tensor of shape [input_dim]
        """
        

        # (1) Apply mask → produce matrix [input_dim, input_dim]
        # mask.unsqueeze(0) -> [1, 10, 10], y.unsqueeze(1) -> [5, 1, 10]
        #   Broadcasting multiplies each vector y[i] with the mask along the last dimension   
        X = self.mask.unsqueeze(0) * y.unsqueeze(1)  # [B, N, N]
        X = X.permute(1, 2, 0)


        # (2) Forward through first layer for all networks at once
        # W1: [N, H, N]
        # X : [N, N, B] 
        h = torch.bmm(self.W1, X) + self.b1.unsqueeze(2)  # [N, H, B]

        h = self.act(h)


        # (3) Forward through hidden layers (vectorized)
        for W, b in zip(self.W_hidden, self.b_hidden):
            # W: [N, H, H], h: [N, H, B]
            h = torch.bmm(W, h) + b.unsqueeze(2)  # [N, H, B]

            h = self.act(h)

        # (4) Output layer
        # W_out: [N, 1, H], h: [N, H, B]
        out = torch.bmm(self.W_out, h) + self.b_out.unsqueeze(2)  # [N, B]

        return y * out.squeeze(1).T  # → [N] transpose because of ealier permutation





class HybridODE(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,         # mechanistic state dimension
        dim_state_latent: int = 10,  # latent state dimension
        use_lotka_volterra: bool = False,
        use_nn_markovian: bool = True,
        use_nn_non_markovian: bool = False,
        num_hidden_layers=3,
        hidden_width=10,
        activation="tanh"
    ):
        super().__init__()

        self.use_lotka_volterra = use_lotka_volterra
        self.use_nn_markovian = use_nn_markovian
        self.use_nn_non_markovian = use_nn_non_markovian

        self.dim_state_mech = input_dim
        self.dim_state_latent = dim_state_latent

        # ---------- Lotka–Volterra ----------
        if use_lotka_volterra:
            self.alpha = nn.Parameter(torch.zeros(input_dim))
            self.A = nn.Linear(input_dim, input_dim, bias=False)
        else:
            self.alpha = None
            self.A = None

        # ---------- Markovian NN ----------
        if use_nn_markovian:
            self.nn_markovian = build_mlp(
                input_dim=input_dim,
                hidden_layers=num_hidden_layers,
                hidden_width=hidden_width,
                output_dim=input_dim,
                activation=activation
            )
        else:
            self.nn_markovian = None

        # ---------- Non-Markovian NN ----------
        if use_nn_non_markovian:
            self.nn_non_markovian = build_mlp(
                input_dim=input_dim + dim_state_latent,
                hidden_layers=num_hidden_layers,
                hidden_width=hidden_width,
                output_dim=input_dim + dim_state_latent,
                activation=activation
            )
        else:
            self.nn_non_markovian = None

        self._initialize_weights()

    # ------------------------------------------------------------
    def _initialize_weights(self):
        # Initialize Markovian NN
        if self.use_nn_markovian:
            for m in self.nn_markovian.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize Non-Markovian NN
        if self.use_nn_non_markovian:
            for m in self.nn_non_markovian.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize LV parameters only if used
        if self.use_lotka_volterra:
            """lotka_weights = torch.load(
                "/home/leo/Documents/MDSINE/NeuralODE/models/model_2integration_lotka_puredyn2_n02.pth"
            )
            self.alpha.data = lotka_weights["alpha"]
            self.A.weight.data = lotka_weights["A.weight"]"""

            nn.init.normal_(self.A.weight, mean=0.0, std=0.001)

            nn.init.normal_(self.alpha, mean=-0.002, std=0.0001)

        
        
    def forward(self, t, y):
        x_lotka = y[:, :self.dim_state_mech]

        dydt = torch.zeros_like(y)

        

        if self.use_lotka_volterra:
            lokta_term = (self.alpha + (self.A(x_lotka))) * x_lotka
            lokta_term = nn.functional.pad(lokta_term, (0, self.dim_state_latent))
            dydt += lokta_term

        if self.use_nn_markovian:
            nn_markovian_term = self.nn_markovian(x_lotka)
            nn_markovian_term = nn.functional.pad(nn_markovian_term, (0, self.dim_state_latent))
            dydt += nn_markovian_term * x_lotka

        if self.use_nn_non_markovian:
            nn_non_markovian_term = self.nn_non_markovian(y)
            dydt += nn_non_markovian_term * y

        return dydt
    




class MicrobTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, device='cpu'):
        self.path = data_path
        self.datasets = []
        self.device = device
        
        for dataset_file in os.listdir(data_path):
            if dataset_file.endswith('.csv'):
                data = pd.read_csv(
                    os.path.join(data_path, dataset_file)
                )
                data_tensor = torch.FloatTensor(data.values).to(device)  # Move to GPU here
                self.datasets.append(data_tensor)

    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        return self.datasets[idx]