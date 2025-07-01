import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

torch.cuda.empty_cache()


SEED = 42
torch.manual_seed(SEED)


class MyNormalization(nn.Module):
    def __init__(self):
        super(MyNormalization, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        std = torch.std(x, dim=0, keepdim=True)
        return (x - mean) / (std + 1e-5)



class SimpleNN(nn.Module):
    def __init__(self, num_layers=7, activation="Sigmoid", init_weights="Default", norm_type=None, use_skip=False):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.norm_type = norm_type
        self.use_skip = use_skip
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.Linear(10, 10))
            if norm_type == "BatchNorm":
                self.norm_layers.append(nn.BatchNorm1d(10))
            elif norm_type == "CustomNorm":
                self.norm_layers.append(MyNormalization())

        self.init_weights(init_weights)

    def init_weights(self, method):
        if method == "Default":
            pass
        elif method == "Increase_std=1.0":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=1.0)
                    nn.init.constant_(module.bias, 0.0)
        elif method == "Increase_std=10.0":
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=10.0)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        skip_connection = x
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if self.norm_type is not None:
                x = self.norm_layers[i](x)

            if self.activation == "Sigmoid":
                x = torch.sigmoid(x)
            elif self.activation == "Tanh":
                x = torch.tanh(x)
            elif self.activation == "ReLU":
                x = F.relu(x)

            if self.use_skip and i % 2 == 1:
                x = x + skip_connection
                skip_connection = x

        return x

st.title("Advanced Gradient Vanishing Demo")

st.sidebar.header("Settings")

num_layers = 7


activation = st.sidebar.selectbox(
    "Activation Function", ["Sigmoid", "Tanh", "ReLU"]
)


init_weights = st.sidebar.selectbox(
    "Weight Initialization", [
        "Default", "Increase_std=1.0", "Increase_std=10.0"]
)


norm_type = st.sidebar.selectbox(
    "Normalization Type", ["None", "BatchNorm", "CustomNorm"]
)


use_skip = st.sidebar.checkbox("Use Skip Connection")

optimizer_type = st.sidebar.selectbox(
    "Optimizer", ["SGD", "Adam"]
)


num_epochs = st.sidebar.slider(
    "Number of Epochs", min_value=1, max_value=100, value=10)


learning_rate = st.sidebar.select_slider(
    "Learning Rate", options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], value=1e-3
)

input_type = "Gaussian (Normal)"


batch_size = 32


if input_type == "Gaussian (Normal)":
    input_data = torch.randn(batch_size, 10)
else:  
    input_data = torch.rand(batch_size, 10)


model = SimpleNN(num_layers=num_layers, activation=activation, init_weights=init_weights,
                 norm_type=norm_type if norm_type != "None" else None, use_skip=use_skip)


if optimizer_type == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_type == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


output = model(input_data)

criterion = nn.MSELoss()  
target = torch.zeros_like(output)  


losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()  
    output = model(input_data)  
    loss = criterion(output, target)  
    loss.backward()  
    optimizer.step()  
    losses.append(loss.item())  

gradients = []
for i, layer in enumerate(model.layers):
    if layer.weight.grad is not None:
        gradients.append(layer.weight.grad.norm().item())
    else:
        gradients.append(0)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=list(range(1, len(gradients) + 1)),
        y=gradients,
        mode="lines+markers+text",
        text=[f"{g:.4f}" for g in gradients],
        textposition="top center",
        name="Gradient Mean",
        line=dict(color="blue"),
    )
)

fig.update_layout(
    title="Gradient Mean Across 7 Layers",
    xaxis_title="Layer",
    yaxis_title="Gradient Mean",
    legend=dict(yanchor="top", y=1, xanchor="right", x=1),
    hovermode="x unified",
)

st.plotly_chart(fig)

st.markdown("### Observations:")
st.markdown(
    f"""
    **Configuration:**
    - **Activation Function:** `{activation}`
    - **Weight Initialization:** `{init_weights}`
    - **Normalization Type:** `{norm_type}`
    - **Optimizer:** `{optimizer_type}`
    - **Learning Rate:** `{learning_rate}`
    - **Skip Connection:** `{"Enabled" if use_skip else "Disabled"}`
    
    **Gradient Insights:**
    - Total Layers: **{num_layers}**
    - Minimum Gradient Mean: **{min(gradients):.4f}**
    - Maximum Gradient Mean: **{max(gradients):.4f}**
    - Gradient Means displayed above for each layer.
    """
)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
         Made by <a>NamNguyen27</a>
    </div>
    """,
    unsafe_allow_html=True
)