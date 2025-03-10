import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HyperbolicTangentSpace(nn.Module):
    """
    Implements operations for the tangent space of the Poincaré ball model of hyperbolic space.
    
    The Poincaré ball model represents hyperbolic space as the interior of the unit ball
    in Euclidean space, where the distance increases exponentially as points approach the boundary.
    """
    
    def __init__(self, curvature: float = 1.0):
        """
        Initialize the hyperbolic tangent space.
        
        Args:
            curvature: Curvature of the hyperbolic space (c > 0)
        """
        super().__init__()
        self.c = nn.Parameter(torch.tensor([curvature]), requires_grad=False)
        
    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the conformal factor.
        
        Args:
            x: Tensor of shape (..., dim) in the Poincaré ball
            
        Returns:
            Conformal factor lambda_c(x)
        """
        c = self.c
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * x_norm_squared).clamp_min(1e-15)
    
    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from the tangent space at the origin to the manifold.
        
        Args:
            v: Tensor of shape (..., dim) in the tangent space at the origin
            
        Returns:
            Points in the Poincaré ball
        """
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        zeros = torch.zeros_like(v_norm)
        
        # Handle zero-norm case
        condition = (v_norm == 0).expand_as(v)
        v_norm = torch.where(condition, torch.ones_like(v_norm), v_norm)
        
        # Apply exponential map
        c = self.c
        sqrt_c = c.sqrt()
        exp_map = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
        
        # Handle zero-norm case
        return torch.where(condition, zeros, exp_map)
    
    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from the manifold to the tangent space at the origin.
        
        Args:
            x: Tensor of shape (..., dim) in the Poincaré ball
            
        Returns:
            Points in the tangent space at the origin
        """
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        zeros = torch.zeros_like(x_norm)
        
        # Handle zero-norm case
        condition = (x_norm == 0).expand_as(x)
        x_norm = torch.where(condition, torch.ones_like(x_norm), x_norm)
        
        # Apply logarithmic map
        c = self.c
        sqrt_c = c.sqrt()
        log_map = torch.atanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)
        
        # Handle zero-norm case
        return torch.where(condition, zeros, log_map)
    
    def mobius_addition(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.
        
        Args:
            x: Tensor of shape (..., dim) in the Poincaré ball
            y: Tensor of shape (..., dim) in the Poincaré ball
            
        Returns:
            Result of the Möbius addition operation
        """
        c = self.c
        x_dot_y = torch.sum(x * y, dim=-1, keepdim=True)
        x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_squared = torch.sum(y * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2*c*x_dot_y + c*y_norm_squared) * x + (1 - c*x_norm_squared) * y
        denominator = 1 + 2*c*x_dot_y + c*c*x_norm_squared*y_norm_squared
        
        return numerator / denominator.clamp_min(1e-15)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points in the Poincaré ball.
        
        Args:
            x: Tensor of shape (..., dim) in the Poincaré ball
            y: Tensor of shape (..., dim) in the Poincaré ball
            
        Returns:
            Tensor of shape (...) containing pairwise distances
        """
        c = self.c
        sqrt_c = c.sqrt()
        
        # Compute Möbius addition -x ⊕ y
        minus_x = -x
        z = self.mobius_addition(minus_x, y)
        
        # Compute distance
        z_norm = torch.norm(z, p=2, dim=-1)
        return 2.0 / sqrt_c * torch.atanh(sqrt_c * z_norm)
    
    def proj(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Project points to ensure they remain in the Poincaré ball.
        
        Args:
            x: Tensor of shape (..., dim) in Euclidean space
            eps: Small constant for numerical stability
            
        Returns:
            Projected tensor in the Poincaré ball
        """
        c = self.c
        
        # Compute norm and maximum allowed norm
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        maxnorm = (1.0 - eps) / (c.sqrt())
        
        # Project if necessary
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)


class HyperbolicLinear(nn.Module):
    """
    Hyperbolic linear layer that operates in the tangent space.
    
    Projects input from hyperbolic space to tangent space, applies a linear transformation,
    and projects back to hyperbolic space.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, curvature: float = 1.0):
        """
        Initialize the hyperbolic linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: Whether to include a bias term
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.hyp = HyperbolicTangentSpace(curvature)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the layer parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Project to tangent space at origin
        x_tan = self.hyp.logmap0(x)
        
        # Apply linear transformation
        output = F.linear(x_tan, self.weight, self.bias)
        
        # Project back to hyperbolic space
        output_hyp = self.hyp.expmap0(output)
        
        # Ensure output is in the Poincaré ball
        output_hyp = self.hyp.proj(output_hyp)
        
        return output_hyp


class HyperbolicActivation(nn.Module):
    """
    Hyperbolic activation function.
    
    Projects input from hyperbolic space to tangent space, applies an activation function,
    and projects back to hyperbolic space.
    """
    
    def __init__(self, activation: nn.Module, curvature: float = 1.0):
        """
        Initialize the hyperbolic activation function.
        
        Args:
            activation: Activation function to apply in tangent space
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        self.activation = activation
        self.hyp = HyperbolicTangentSpace(curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Project to tangent space at origin
        x_tan = self.hyp.logmap0(x)
        
        # Apply activation
        output = self.activation(x_tan)
        
        # Project back to hyperbolic space
        output_hyp = self.hyp.expmap0(output)
        
        # Ensure output is in the Poincaré ball
        output_hyp = self.hyp.proj(output_hyp)
        
        return output_hyp


class HyperbolicGRU(nn.Module):
    """
    Hyperbolic Gated Recurrent Unit (GRU) for temporal sequences.
    
    Processes sequences in hyperbolic space for dynamic graph embeddings.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, curvature: float = 1.0):
        """
        Initialize the hyperbolic GRU.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hyp = HyperbolicTangentSpace(curvature)
        
        # GRU gates in Euclidean space
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.h_tilde = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor in hyperbolic space (batch_size, input_dim)
            h: Hidden state in hyperbolic space (batch_size, hidden_dim)
            
        Returns:
            Updated hidden state in hyperbolic space
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            # Ensure it's in the Poincaré ball
            h = self.hyp.proj(h)
        
        # Project to tangent space
        x_tan = self.hyp.logmap0(x)
        h_tan = self.hyp.logmap0(h)
        
        # Concatenate input and hidden state
        xh = torch.cat([x_tan, h_tan], dim=1)
        
        # Calculate gate values in Euclidean space
        r = torch.sigmoid(self.reset_gate(xh))
        z = torch.sigmoid(self.update_gate(xh))
        
        # Calculate candidate hidden state
        xh_reset = torch.cat([x_tan, r * h_tan], dim=1)
        h_tilde_tan = torch.tanh(self.h_tilde(xh_reset))
        
        # Update hidden state in tangent space
        h_new_tan = (1 - z) * h_tan + z * h_tilde_tan
        
        # Project back to hyperbolic space
        h_new = self.hyp.expmap0(h_new_tan)
        
        # Ensure output is in the Poincaré ball
        h_new = self.hyp.proj(h_new)
        
        return h_new


class EuclideanToHyperbolic(nn.Module):
    """
    Maps Euclidean vectors to the hyperbolic space.
    """
    
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0):
        """
        Initialize the Euclidean to hyperbolic mapping.
        
        Args:
            in_features: Size of each input Euclidean sample
            out_features: Size of each output hyperbolic sample
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.hyp = HyperbolicTangentSpace(curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor in Euclidean space
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Linear mapping in Euclidean space
        h = self.linear(x)
        
        # Map to hyperbolic space
        h_hyp = self.hyp.expmap0(h)
        
        # Ensure output is in the Poincaré ball
        h_hyp = self.hyp.proj(h_hyp)
        
        return h_hyp


class HyperbolicToEuclidean(nn.Module):
    """
    Maps hyperbolic vectors to Euclidean space.
    """
    
    def __init__(self, in_features: int, out_features: int, curvature: float = 1.0):
        """
        Initialize the hyperbolic to Euclidean mapping.
        
        Args:
            in_features: Size of each input hyperbolic sample
            out_features: Size of each output Euclidean sample
            curvature: Curvature of the hyperbolic space
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.hyp = HyperbolicTangentSpace(curvature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in Euclidean space
        """
        # Map to tangent space
        x_tan = self.hyp.logmap0(x)
        
        # Linear mapping in Euclidean space
        return self.linear(x_tan) 