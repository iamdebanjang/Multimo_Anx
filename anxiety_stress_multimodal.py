import math, random, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- utility blocks ---------------------------- #

class SimpleEncoder(nn.Module):
    """1-layer projection + dropout (replace with real models)."""
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid_dim)
        self.var_head = nn.Linear(in_dim, 1)  # log-variance for UAF
    def forward(self, x):
        h = F.relu(self.proj(x.mean(dim=1)))   # (B, hid_dim)
        log_var = self.var_head(x.mean(dim=1)) # (B,1)
        return h, torch.exp(log_var)           # variance (positive)

class VAEBottleneck(nn.Module):
    def __init__(self, hid_dim: int, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(hid_dim, z_dim)
        self.logvar = nn.Linear(hid_dim, z_dim)
    def forward(self, h):
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return z, kld.mean()

class DiffusionImputerStub(nn.Module):
    """Returns zeros for missing modalities – replace later."""
    def forward(self, x_avail, target_shape):
        return torch.zeros(target_shape, device=x_avail.device)

class BayesianHead(nn.Module):
    """Bayesian linear head with closed-form posterior (conjugate)."""
    def __init__(self, in_dim, num_classes, prior_var=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, in_dim))
        self.bias   = nn.Parameter(torch.zeros(num_classes))
        # simple diagonal precision matrix for demonstration
        self.register_buffer("precision", torch.eye(in_dim) / prior_var)
    def forward(self, h):
        return F.log_softmax(F.linear(h, self.weight, self.bias), dim=-1)

# ------------------------------ main model ------------------------------ #

class MultimodalStressNet(nn.Module):
    def __init__(self,
                 audio_dim=128, visual_dim=2048, text_dim=768,
                 physio_dim=32, hid_dim=256, z_dim=128,
                 num_classes=3, use_vae=True):
        super().__init__()
        self.use_physio = True                      # toggle at runtime
        self.use_vae = use_vae

        # encoders
        self.enc_audio  = SimpleEncoder(audio_dim,  hid_dim)
        self.enc_visual = SimpleEncoder(visual_dim,hid_dim)
        self.enc_text   = SimpleEncoder(text_dim,  hid_dim)
        self.enc_physio = SimpleEncoder(physio_dim,hid_dim)

        # attention parameters
        self.W_q = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_k = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W_v = nn.Linear(hid_dim, hid_dim, bias=False)

        # generative bottleneck
        if use_vae:
            self.vae = VAEBottleneck(hid_dim, z_dim)
            self.decoder = nn.ModuleDict({
                'audio':  nn.Linear(z_dim, audio_dim),
                'visual': nn.Linear(z_dim, visual_dim),
                'text':   nn.Linear(z_dim, text_dim),
                'physio': nn.Linear(z_dim, physio_dim)
            })
        self.imputer = DiffusionImputerStub()

        # classifier / regressor
        head_in = z_dim if use_vae else hid_dim
        self.head = BayesianHead(head_in, num_classes)

    # -------------------- fusion helpers -------------------- #
    def _uaf(self, H, var):   # H:(B,M,D)  var:(B,M,1)
        """Uncertainty-Aware Attention Fusion."""
        Q = self.W_q(H)                       # (B,M,D)
        K = self.W_k(H)
        V = self.W_v(H)
        attn_logits = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(H.size(-1))
        alpha = F.softmax(attn_logits, dim=-1)  # (B,M,M)
        H_tilde = torch.matmul(alpha, V)        # (B,M,D)
        # weight each modality by 1/variance
        weights = (1.0 / var.clamp(min=1e-6))   # (B,M,1)
        H_weighted = (H_tilde * weights).sum(dim=1) / weights.sum(dim=1)
        return H_weighted   # fused (B,D)

    # ------------------------ forward ------------------------ #
    def forward(self, batch):
        """
        batch dict keys:
          audio  : (B, T_a, audio_dim)
          visual : (B, T_v, visual_dim)
          text   : (B, T_t, text_dim)
          physio : OPTIONAL (B, T_p, physio_dim)  or None
        """
        # Encode each available modality
        h_list, var_list = [], []
        for key, enc in (('audio', self.enc_audio),
                         ('visual', self.enc_visual),
                         ('text',   self.enc_text)):
            h, var = enc(batch[key])
            h_list.append(h); var_list.append(var)

        # handle physio
        if batch.get('physio') is not None and self.use_physio:
            h_p, var_p = self.enc_physio(batch['physio'])
        else:
            # impute missing physio
            fake = self.imputer(batch['audio'],  # seed tensor for device
                                target_shape=(batch['audio'].size(0), 10, 32))
            h_p, var_p = self.enc_physio(fake)
        h_list.append(h_p); var_list.append(var_p)

        H   = torch.stack(h_list,  dim=1)   # (B,M,D)
        var = torch.stack(var_list, dim=1)  # (B,M,1)

        h_fused = self._uaf(H, var)         # (B,D)

        # VAE bottleneck (optional)
        recon_loss = torch.tensor(0.0, device=h_fused.device)
        kld_loss   = torch.tensor(0.0, device=h_fused.device)
        if self.use_vae:
            z, kld_loss = self.vae(h_fused)
            h_pred = z
            # very simple recon loss on mean features
            for m, lin in self.decoder.items():
                target = batch[m].mean(dim=1)
                recon  = lin(z)
                recon_loss += F.mse_loss(recon, target)
        else:
            h_pred = h_fused

        log_probs = self.head(h_pred)       # (B,num_classes)
        return log_probs, recon_loss, kld_loss


# --------------------------- training --------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    B = 4    # batch size
    T = 16   # timesteps
    model = MultimodalStressNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):  # training loop
        # fake batch ----------------------------------------------------
        batch = {
            'audio' :  torch.randn(B, T, 128),
            'visual':  torch.randn(B, T, 2048),
            'text'  :  torch.randn(B, T, 768),
            'physio':  None if random.random() < 0.5 else torch.randn(B, T, 32)
        }
        labels = torch.randint(0, 3, (B,))

        # forward / loss ------------------------------------------------
        logp, recon, kld = model(batch)
        clf_loss = F.nll_loss(logp, labels)
        loss = clf_loss + 0.1 * recon + 0.01 * kld

        # backward ------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step:02d} | total {loss.item():.3f} | "
              f"clf {clf_loss.item():.3f} | recon {recon.item():.3f} | "
              f"kld {kld.item():.3f}")
