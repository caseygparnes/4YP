"""
Linear Recurrent Unit (LRU) implementation for MAD policy magnitude term.
Based on the MAD paper (Furieri et al., 2025) and LRU design from Orvieto et al. (2023).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from .util import init


class LRU(nn.Module):
    """
    Linear Recurrent Unit for the magnitude term in MAD policies.

    The LRU is a stable recurrent layer with diagonal state transition matrix Λ
    where |λ_i| < 1 for all eigenvalues λ_i, ensuring stability.

    Architecture:
        ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
        M_t = Re(C ξ_t) + D v_t + F v_t

    where:
        - ξ_t ∈ C^{n_hidden} is the internal state (complex-valued)
        - Λ is a diagonal matrix with |λ_i| < 1
        - Γ(Λ) is a normalization term
        - v_t is the input (x_0 at t=0, then 0)
        - Re(·) extracts the real part
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_orthogonal: bool = True,
        gain: float = 0.01,
    ):
        """
        Args:
            input_dim: Dimension of input (state dimension)
            hidden_dim: Dimension of hidden state (complex-valued)
            output_dim: Dimension of output (action dimension)
            use_orthogonal: Whether to use orthogonal initialization
            gain: Gain for initialization
        """
        super(LRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize diagonal complex-valued state transition matrix Λ
        # We parametrize it using phase and magnitude to ensure |λ_i| < 1
        # λ_i = r_i * exp(iθ_i) where r_i ∈ (0, 1)
        self.log_r = nn.Parameter(torch.randn(hidden_dim) * 0.1 - 1.0)  # log(r) to ensure r > 0
        self.theta = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # Input matrix B (maps input to hidden state)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)

        # Output matrix C (maps hidden state to output, applied to real part)
        self.C = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

        # Feedthrough matrices D and F
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.F = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        # Normalization for stability
        self.gamma_log = nn.Parameter(torch.zeros(hidden_dim))

    def get_lambda(self):
        """Get the diagonal state transition matrix values."""
        # Ensure |λ| < 1 by using sigmoid on r
        r = torch.sigmoid(self.log_r)
        # λ = r * exp(iθ)
        lambda_real = r * torch.cos(self.theta)
        lambda_imag = r * torch.sin(self.theta)
        return lambda_real, lambda_imag

    def get_gamma(self):
        """Get normalization factor Γ(Λ)."""
        # Γ is typically sqrt(1 - |λ|^2) for normalization
        r = torch.sigmoid(self.log_r)
        return torch.sqrt(1 - r**2 + 1e-8)

    def forward(
        self,
        x0: torch.Tensor,
        seq_len: int,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LRU.

        Args:
            x0: Initial condition [batch_size, input_dim]
            seq_len: Number of timesteps to unroll
            hidden_state: Optional previous hidden state [batch_size, hidden_dim, 2]
                         (real and imaginary parts)

        Returns:
            output: Magnitude values for each timestep [batch_size, seq_len, output_dim]
            final_hidden: Final hidden state [batch_size, hidden_dim, 2]
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Initialize hidden state
        if hidden_state is None:
            # Initialize with zeros (complex)
            hidden_real = torch.zeros(batch_size, self.hidden_dim, device=device)
            hidden_imag = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            hidden_real = hidden_state[:, :, 0]
            hidden_imag = hidden_state[:, :, 1]

        # Get lambda and gamma
        lambda_real, lambda_imag = self.get_lambda()
        gamma = self.get_gamma()

        outputs = []

        for t in range(seq_len):
            # Input at time t: x_0 if t=0, else 0
            if t == 0:
                v_t = x0
            else:
                v_t = torch.zeros_like(x0)

            # Compute Γ(Λ) B v_t
            input_contribution = torch.matmul(v_t, self.B.t())  # [batch_size, hidden_dim]
            input_contribution = gamma.unsqueeze(0) * input_contribution

            # Update hidden state: ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
            # Complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
            new_hidden_real = lambda_real * hidden_real - lambda_imag * hidden_imag + input_contribution
            new_hidden_imag = lambda_real * hidden_imag + lambda_imag * hidden_real

            hidden_real = new_hidden_real
            hidden_imag = new_hidden_imag

            # Compute output: M_t = Re(C ξ_t) + D v_t + F v_t
            # Extract real part of hidden state
            output_from_hidden = torch.matmul(hidden_real, self.C.t())  # [batch_size, output_dim]
            output_from_input = torch.matmul(v_t, (self.D + self.F).t())

            output = output_from_hidden + output_from_input
            outputs.append(output)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, output_dim]

        # Return final hidden state
        final_hidden = torch.stack([hidden_real, hidden_imag], dim=-1)  # [batch_size, hidden_dim, 2]

        return outputs, final_hidden

    def step(
        self,
        v_t: torch.Tensor,
        hidden_state: torch.Tensor,
        is_first_step = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step forward (for online execution).

        Args:
            v_t: Input at current timestep [batch_size, input_dim]
                 (should be x_0 if is_first_step=True, else can be anything but typically 0)
            hidden_state: Current hidden state [batch_size, hidden_dim, 2]
            is_first_step: Whether this is the first step (uses v_t as x_0)
                          Can be either:
                          - bool: scalar for all environments
                          - torch.Tensor: [batch_size] boolean tensor for per-environment control

        Returns:
            output: Magnitude value [batch_size, output_dim]
            new_hidden: Updated hidden state [batch_size, hidden_dim, 2]
        """
        hidden_real = hidden_state[:, :, 0]
        hidden_imag = hidden_state[:, :, 1]

        # Get lambda and gamma
        lambda_real, lambda_imag = self.get_lambda()
        gamma = self.get_gamma()

        # Handle per-environment or scalar first-step detection
        # v_t should already be set correctly by the caller (x0 or zeros)
        # The caller uses torch.where to set v_t based on is_first_step
        # So we don't need to modify v_t here anymore

        # Compute Γ(Λ) B v_t
        input_contribution = torch.matmul(v_t, self.B.t())  # [batch_size, hidden_dim]
        input_contribution = gamma.unsqueeze(0) * input_contribution

        # Update hidden state: ξ_{t+1} = Λ ξ_t + Γ(Λ) B v_t
        new_hidden_real = lambda_real * hidden_real - lambda_imag * hidden_imag + input_contribution
        new_hidden_imag = lambda_real * hidden_imag + lambda_imag * hidden_real

        # Compute output: M_t = Re(C ξ_t) + D v_t + F v_t
        output_from_hidden = torch.matmul(new_hidden_real, self.C.t())
        output_from_input = torch.matmul(v_t, (self.D + self.F).t())

        output = output_from_hidden + output_from_input

        # Return new hidden state
        new_hidden = torch.stack([new_hidden_real, new_hidden_imag], dim=-1)

        return output, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_dim, 2, device=device)
