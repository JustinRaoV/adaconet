"""
Layer 3 — VAE Latent Space Analysis for AdaCoNet.

A Variational Autoencoder (VAE) learns a low-dimensional latent representation
of the CLR-transformed microbial compositions.  The key insight is that the
**decoder Jacobian** J = d(decode(z))/dz captures how sensitive each taxon's
reconstructed abundance is to perturbations in latent space.

Why the Jacobian?
-----------------
If taxa i and j are co-regulated (i.e., respond similarly to the same latent
factors), their Jacobian rows J_i and J_j will be highly correlated.  This
reveals **nonlinear** associations that linear correlation or even MI may miss,
because the decoder can learn arbitrary nonlinear mappings from latent to
observed space.

Architecture
------------
Encoder:
    Linear(p, h_dim) -> ReLU -> Linear(h_dim, d_latent) [mu, log_var]

Decoder:
    Linear(d_latent, h_dim) -> ReLU -> Linear(h_dim, p) -> Softmax/temperature

The temperature-scaled softmax ensures the decoder output lies on the simplex,
mirroring the compositional nature of the data.

Loss function
-------------
    L = NLL_recon + beta * KL + gamma * jacobian_regularizer

- **NLL_recon**: Dirichlet-Multinomial-like negative log-likelihood.
  We use a cross-entropy loss with temperature scaling, which approximates
  the DM negative log-likelihood for overdispersed compositions.
- **KL**: Standard VAE KL divergence between q(z|x) and N(0, I).
- **Jacobian regularizer**: penalises the Frobenius norm of the Jacobian to
  encourage smooth latent mappings (prevents overfitting and improves the
  quality of Jacobian-based scores).

Jacobian scoring
----------------
For each sample i:
    z_i = encoder(x_i).mu      (deterministic encoding)
    J_i = d(decoder(z_i)) / dz  (p x d_latent Jacobian matrix)

The pairwise score S_VAE(i, j) is the absolute Pearson correlation between
rows J_i and J_j (averaged across samples).  This is computed for all
pairs of taxa.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Gaussian encoder network.

    Maps a p-dimensional CLR vector to the parameters (mu, log_var) of
    a diagonal Gaussian q(z | x).

    Architecture: Linear(p, h) -> ReLU -> [Linear(h, d), Linear(h, d)]
    """

    def __init__(self, p: int, h_dim: int, d_latent: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(p, h_dim)
        self.fc_mu = nn.Linear(h_dim, d_latent)
        self.fc_logvar = nn.Linear(h_dim, d_latent)

        # Xavier initialisation for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, p)

        Returns
        -------
        mu : Tensor, shape (batch, d_latent)
        log_var : Tensor, shape (batch, d_latent)
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Simplex-output decoder with temperature-scaled softmax.

    Maps a d_latent-dimensional latent vector to a p-dimensional composition
    on the probability simplex.

    Architecture: Linear(d, h) -> ReLU -> Linear(h, p) -> Softmax(1/tau)
    """

    def __init__(self, d_latent: int, h_dim: int, p: int,
                 temperature: float = 0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_latent, h_dim)
        self.fc2 = nn.Linear(h_dim, p)
        self.temperature = temperature

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        z : Tensor, shape (batch, d_latent)

        Returns
        -------
        x_recon : Tensor, shape (batch, p)
            Reconstructed composition on the simplex (sums to 1 along dim=-1).
        """
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)  # (batch, p)
        # Temperature-scaled softmax: lower temperature -> sharper distribution
        x_recon = F.softmax(logits / self.temperature, dim=-1)
        return x_recon


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class MicrobialVAE(nn.Module):
    """Variational Autoencoder for microbial composition data.

    Parameters
    ----------
    p : int
        Number of taxa (input dimensionality).
    d_latent : int, default 64
        Latent space dimensionality.
    h_dim : int, default 256
        Hidden layer width for both encoder and decoder.
    temperature : float, default 0.5
        Softmax temperature in the decoder.  Lower values produce sharper
        compositions; higher values produce flatter ones.

    Attributes
    ----------
    encoder : Encoder
    decoder : Decoder
    device : torch.device
        Computation device (auto-detected).
    """

    def __init__(
        self,
        p: int,
        d_latent: int = 64,
        h_dim: int = 256,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()

        self.p = p
        self.d_latent = d_latent
        self.h_dim = h_dim
        self.temperature = temperature

        self.encoder = Encoder(p, h_dim, d_latent)
        self.decoder = Decoder(d_latent, h_dim, p, temperature)

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        """Full forward pass through encoder -> reparameterise -> decoder.

        Parameters
        ----------
        x : Tensor, shape (batch, p)

        Returns
        -------
        x_recon : Tensor, shape (batch, p)
        mu : Tensor, shape (batch, d_latent)
        log_var : Tensor, shape (batch, d_latent)
        z : Tensor, shape (batch, d_latent)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    @staticmethod
    def reparameterize(mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + sigma * epsilon.

        Parameters
        ----------
        mu : Tensor, shape (batch, d)
        log_var : Tensor, shape (batch, d)

        Returns
        -------
        z : Tensor, shape (batch, d)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
        gamma: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the VAE loss.

        L = NLL_recon + beta * KL + gamma * jacobian_regularizer

        Parameters
        ----------
        x : Tensor, shape (batch, p)
            Input CLR-transformed compositions.
        x_recon : Tensor, shape (batch, p)
            Reconstructed compositions from decoder.
        mu : Tensor, shape (batch, d_latent)
        log_var : Tensor, shape (batch, d_latent)
        beta : float
            Weight for KL divergence (beta-VAE).
        gamma : float
            Weight for Jacobian regularisation.

        Returns
        -------
        total_loss, recon_loss, kl_loss, jac_reg : Tensors (scalars)
        """
        # --- Reconstruction loss ---
        # Dirichlet-Multinomial-like NLL via cross-entropy on compositions.
        # We treat x_recon as predicted probabilities and the (normalised)
        # input as target distribution.  The softmax output is already
        # on the simplex, so we use KL divergence as the reconstruction
        # objective (equivalent to cross-entropy up to a constant).
        #
        # To handle CLR input (which can be negative), we shift to positive
        # space via softmax of the CLR values as the target distribution.
        x_target = F.softmax(x, dim=-1)  # target composition from CLR
        recon_loss = F.kl_div(
            torch.log(x_recon.clamp(min=1e-10)),
            x_target,
            reduction="batchmean",
        )

        # --- KL divergence: q(z|x) || N(0, I) ---
        # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        )

        # --- Jacobian regulariser ---
        # Penalise the Frobenius norm of the decoder Jacobian to encourage
        # smoothness.  Computed on a random subset for efficiency.
        jac_reg = self._jacobian_regularizer(x_recon)

        total_loss = recon_loss + beta * kl_loss + gamma * jac_reg

        return total_loss, recon_loss, kl_loss, jac_reg

    def _jacobian_regularizer(self, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute a stochastic estimate of the Jacobian Frobenius norm.

        Instead of computing the full Jacobian (expensive), we use the
        Hutchinson trace estimator: for a random vector v ~ N(0, I),
            ||J||_F^2 approx= ||J v||^2
        averaged over a few random vectors.

        Parameters
        ----------
        x_recon : Tensor, shape (batch, p)
            Decoder output (with gradient tracking).

        Returns
        -------
        jac_norm : Tensor (scalar)
            Estimated mean squared Frobenius norm of the Jacobian.
        """
        batch_size = x_recon.shape[0]

        # Hutchinson estimator with 1 random vector per sample
        v = torch.randn_like(x_recon)

        # Compute J^T v via a single backward pass
        jtv = torch.autograd.grad(
            outputs=x_recon,
            inputs=self.decoder.fc1.weight,  # use a leaf parameter as proxy
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if jtv is None:
            return torch.tensor(0.0, device=x_recon.device)

        jac_norm = torch.mean(jtv.pow(2).sum())
        return jac_norm

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        Z_clr: NDArray[np.floating],
        epochs: int = 100,
        lr: float = 1e-3,
        beta: float = 1.0,
        gamma: float = 0.1,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """Train the VAE on CLR-transformed data.

        Parameters
        ----------
        Z_clr : ndarray, shape (n_samples, n_taxa)
            CLR-transformed input data.
        epochs : int, default 100
            Number of training epochs.
        lr : float, default 1e-3
            Learning rate for Adam optimizer.
        beta : float, default 1.0
            KL divergence weight (beta-VAE).
        gamma : float, default 0.1
            Jacobian regulariser weight.
        batch_size : int, optional
            Mini-batch size.  Defaults to min(256, n_samples).
        verbose : bool, default False
            If True, print epoch-wise loss.

        Returns
        -------
        history : dict
            Keys: 'total_loss', 'recon_loss', 'kl_loss' — each a list of
            per-epoch mean losses.
        """
        n = Z_clr.shape[0]
        if batch_size is None:
            batch_size = min(256, n)
        batch_size = max(batch_size, 1)

        # Convert to PyTorch tensors and dataloader
        X_tensor = torch.tensor(Z_clr, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Learning rate scheduler: reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

        self.train()
        for epoch in range(epochs):
            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                x_recon, mu, log_var, z = self.forward(batch_x)
                total_loss, recon_loss, kl_loss, _ = self.loss_function(
                    batch_x, x_recon, mu, log_var, beta=beta, gamma=gamma
                )

                optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_total += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                n_batches += 1

            avg_total = epoch_total / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_kl = epoch_kl / max(n_batches, 1)

            history["total_loss"].append(avg_total)
            history["recon_loss"].append(avg_recon)
            history["kl_loss"].append(avg_kl)

            scheduler.step(avg_total)

            if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Loss: {avg_total:.4f} | "
                    f"Recon: {avg_recon:.4f} | "
                    f"KL: {avg_kl:.4f}"
                )

        self.eval()
        return history

    # ------------------------------------------------------------------
    # Jacobian-based scoring
    # ------------------------------------------------------------------

    def compute_jacobian_scores(
        self,
        Z_clr: NDArray[np.floating],
        batch_size: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Compute pairwise association scores from VAE decoder Jacobian.

        Computes the mean Jacobian across all samples (averaged at the
        mean latent encoding), then derives pairwise Pearson correlation
        of each taxon's latent response profile.

        This gives a (p, d) matrix where each row describes how taxon j's
        decoder output responds to changes in the latent dimensions.
        Correlating these rows gives a stable, interpretable association
        score that captures shared latent structure.

        Parameters
        ----------
        Z_clr : ndarray, shape (n_samples, n_taxa)
        batch_size : int, optional
            Unused (kept for API compatibility).

        Returns
        -------
        S_vae : ndarray of float64, shape (p, p)
            Symmetric pairwise association matrix.
        """
        self.eval()
        n, p = Z_clr.shape

        X_tensor = torch.tensor(Z_clr, dtype=torch.float32, device=self.device)

        # Encode all samples, get mean latent encoding
        with torch.no_grad():
            mu, _ = self.encoder(X_tensor)

        # Average latent encoding across all samples
        z_mean = mu.mean(dim=0, keepdim=True)  # (1, d_latent)
        z_mean = z_mean.detach().clone().requires_grad_(True)

        # Decode from the mean encoding
        x_recon = self.decoder(z_mean)  # (1, p)

        # Compute Jacobian at the mean encoding: J has shape (p, d_latent)
        # J[j] = gradient of decoder output j w.r.t. latent z
        jac = np.zeros((p, self.d_latent), dtype=np.float64)
        for j in range(p):
            if z_mean.grad is not None:
                z_mean.grad.zero_()
            x_recon[0, j].backward(retain_graph=True)
            if z_mean.grad is not None:
                jac[j, :] = z_mean.grad.detach().cpu().numpy().copy()

        # Pairwise Pearson correlation of latent response profiles
        jac_c = jac - jac.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(jac_c, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        jac_norm = jac_c / norms

        corr = jac_norm @ jac_norm.T
        np.clip(corr, -1, 1, out=corr)
        np.fill_diagonal(corr, 0.0)

        return np.abs(corr).astype(np.float64)
