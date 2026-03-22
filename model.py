import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _similarity_terms(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        denom = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        denom = denom.clamp_min(1e-15)
        return refl_sim, between_sim, denom

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                  between_pos_mask: torch.Tensor = None):
        _, between_sim, denom = self._similarity_terms(z1, z2)

        if between_pos_mask is None:
            numerator = between_sim.diag()
        else:
            pos_mask = between_pos_mask.float()
            numerator = (between_sim * pos_mask).sum(1)
            numerator = numerator.clamp_min(1e-15)

        return -torch.log(numerator / denom)

    def corrected_semi_loss(self,
                            z1: torch.Tensor,
                            z2: torch.Tensor,
                            du_pos_mask: torch.Tensor,
                            du_pos_weight: torch.Tensor,
                            unlabeled_weight: float = 1.0):
        _, between_sim, denom = self._similarity_terms(z1, z2)

        # D_L^+: original augmented positive pairs on the diagonal.
        dl_loss = -torch.log(between_sim.diag().clamp_min(1e-15) / denom)

        # D_U^+: mined unlabeled positives with exponential weights.
        prob_matrix = between_sim / denom.unsqueeze(1)
        prob_matrix = prob_matrix.clamp_min(1e-15)
        weighted_nll = du_pos_weight * (-torch.log(prob_matrix))

        du_weight_sum = du_pos_weight.sum(1)
        has_du = du_weight_sum > 0
        du_loss = torch.zeros_like(dl_loss)
        du_loss[has_du] = weighted_nll[has_du].sum(1) / du_weight_sum[has_du]

        return dl_loss + unlabeled_weight * du_loss

    def corrected_semi_loss_iflgc(self,
                                  z1: torch.Tensor,
                                  z2: torch.Tensor,
                                  du_pos_mask: torch.Tensor,
                                  du_pos_weight: torch.Tensor,
                                  unlabeled_weight: float = 1.0,
                                  refl_du_weight: float = 0.3):
        refl_sim, between_sim, denom = self._similarity_terms(z1, z2)

        # D_L^+: original augmented positive pairs on the diagonal.
        dl_loss = -torch.log(between_sim.diag().clamp_min(1e-15) / denom)

        # D_U^+: semantically guided positives contribute from both views.
        prob_between = (between_sim / denom.unsqueeze(1)).clamp_min(1e-15)
        prob_refl = (refl_sim / denom.unsqueeze(1)).clamp_min(1e-15)

        weighted_nll_between = du_pos_weight * (-torch.log(prob_between))
        weighted_nll_refl = du_pos_weight * (-torch.log(prob_refl))

        du_weight_sum = du_pos_weight.sum(1)
        has_du = du_weight_sum > 0

        du_loss_between = torch.zeros_like(dl_loss)
        du_loss_refl = torch.zeros_like(dl_loss)
        du_loss_between[has_du] = weighted_nll_between[has_du].sum(1) / du_weight_sum[has_du]
        du_loss_refl[has_du] = weighted_nll_refl[has_du].sum(1) / du_weight_sum[has_du]

        refl_du_weight = min(max(float(refl_du_weight), 0.0), 1.0)
        du_loss = (1.0 - refl_du_weight) * du_loss_between + refl_du_weight * du_loss_refl

        return dl_loss + unlabeled_weight * du_loss

    def batched_corrected_semi_loss(self,
                                    z1: torch.Tensor,
                                    z2: torch.Tensor,
                                    du_pos_weight: torch.Tensor,
                                    batch_size: int,
                                    unlabeled_weight: float = 1.0):
        # Space complexity: O(BN) for corrected IFL-GR variant.
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            local_idx = torch.arange(mask.size(0), device=device)
            refl_diag = refl_sim[local_idx, mask]
            between_diag = between_sim[local_idx, mask].clamp_min(1e-15)

            denom = (refl_sim.sum(1) + between_sim.sum(1) - refl_diag).clamp_min(1e-15)
            dl_loss = -torch.log(between_diag / denom)

            prob_between = (between_sim / denom.unsqueeze(1)).clamp_min(1e-15)
            batch_weight = du_pos_weight[mask].float()
            weighted_nll = batch_weight * (-torch.log(prob_between))

            du_weight_sum = batch_weight.sum(1)
            has_du = du_weight_sum > 0
            du_loss = torch.zeros_like(dl_loss)
            if has_du.any():
                du_loss[has_du] = weighted_nll[has_du].sum(1) / du_weight_sum[has_du]

            losses.append(dl_loss + unlabeled_weight * du_loss)

        return torch.cat(losses)

    def batched_corrected_semi_loss_iflgc(self,
                                          z1: torch.Tensor,
                                          z2: torch.Tensor,
                                          du_pos_weight: torch.Tensor,
                                          batch_size: int,
                                          unlabeled_weight: float = 1.0,
                                          refl_du_weight: float = 0.3):
        # Space complexity: O(BN) for corrected IFL-GC variant.
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []
        refl_du_weight = min(max(float(refl_du_weight), 0.0), 1.0)

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            local_idx = torch.arange(mask.size(0), device=device)
            refl_diag = refl_sim[local_idx, mask]
            between_diag = between_sim[local_idx, mask].clamp_min(1e-15)

            denom = (refl_sim.sum(1) + between_sim.sum(1) - refl_diag).clamp_min(1e-15)
            dl_loss = -torch.log(between_diag / denom)

            prob_between = (between_sim / denom.unsqueeze(1)).clamp_min(1e-15)
            prob_refl = (refl_sim / denom.unsqueeze(1)).clamp_min(1e-15)

            batch_weight = du_pos_weight[mask].float()
            weighted_nll_between = batch_weight * (-torch.log(prob_between))
            weighted_nll_refl = batch_weight * (-torch.log(prob_refl))

            du_weight_sum = batch_weight.sum(1)
            has_du = du_weight_sum > 0

            du_loss_between = torch.zeros_like(dl_loss)
            du_loss_refl = torch.zeros_like(dl_loss)
            if has_du.any():
                du_loss_between[has_du] = weighted_nll_between[has_du].sum(1) / du_weight_sum[has_du]
                du_loss_refl[has_du] = weighted_nll_refl[has_du].sum(1) / du_weight_sum[has_du]

            du_loss = (1.0 - refl_du_weight) * du_loss_between + refl_du_weight * du_loss_refl
            losses.append(dl_loss + unlabeled_weight * du_loss)

        return torch.cat(losses)

    def batched_corrected_semi_loss_sparse(self,
                                           z1: torch.Tensor,
                                           z2: torch.Tensor,
                                           du_row_ptr: torch.Tensor,
                                           du_col_idx: torch.Tensor,
                                           du_col_w: torch.Tensor,
                                           batch_size: int,
                                           unlabeled_weight: float = 1.0):
        # Sparse DU+ version for IFL-GR: O(BN + B*K), avoids dense NxN DU weights.
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            local_idx = torch.arange(mask.size(0), device=device)
            refl_diag = refl_sim[local_idx, mask]
            between_diag = between_sim[local_idx, mask].clamp_min(1e-15)

            denom = (refl_sim.sum(1) + between_sim.sum(1) - refl_diag).clamp_min(1e-15)
            dl_loss = -torch.log(between_diag / denom)

            prob_between = (between_sim / denom.unsqueeze(1)).clamp_min(1e-15)
            du_loss = torch.zeros_like(dl_loss)

            for bi in range(mask.size(0)):
                r = int(mask[bi].item())
                s = int(du_row_ptr[r].item())
                e = int(du_row_ptr[r + 1].item())
                if e <= s:
                    continue

                cols = du_col_idx[s:e]
                ws = du_col_w[s:e]
                probs = prob_between[bi, cols]
                du_loss[bi] = (ws * (-torch.log(probs))).sum() / ws.sum().clamp_min(1e-15)

            losses.append(dl_loss + unlabeled_weight * du_loss)

        return torch.cat(losses)

    def batched_corrected_semi_loss_iflgc_sparse(self,
                                                 z1: torch.Tensor,
                                                 z2: torch.Tensor,
                                                 du_row_ptr: torch.Tensor,
                                                 du_col_idx: torch.Tensor,
                                                 du_col_w: torch.Tensor,
                                                 batch_size: int,
                                                 unlabeled_weight: float = 1.0,
                                                 refl_du_weight: float = 0.3):
        # Sparse DU+ version for IFL-GC: O(BN + B*K), avoids dense NxN DU weights.
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes, device=device)
        losses = []
        refl_du_weight = min(max(float(refl_du_weight), 0.0), 1.0)

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))
            between_sim = f(self.sim(z1[mask], z2))

            local_idx = torch.arange(mask.size(0), device=device)
            refl_diag = refl_sim[local_idx, mask]
            between_diag = between_sim[local_idx, mask].clamp_min(1e-15)

            denom = (refl_sim.sum(1) + between_sim.sum(1) - refl_diag).clamp_min(1e-15)
            dl_loss = -torch.log(between_diag / denom)

            prob_between = (between_sim / denom.unsqueeze(1)).clamp_min(1e-15)
            prob_refl = (refl_sim / denom.unsqueeze(1)).clamp_min(1e-15)

            du_loss_between = torch.zeros_like(dl_loss)
            du_loss_refl = torch.zeros_like(dl_loss)

            for bi in range(mask.size(0)):
                r = int(mask[bi].item())
                s = int(du_row_ptr[r].item())
                e = int(du_row_ptr[r + 1].item())
                if e <= s:
                    continue

                cols = du_col_idx[s:e]
                ws = du_col_w[s:e]
                pb = prob_between[bi, cols]
                pr = prob_refl[bi, cols]

                wsum = ws.sum().clamp_min(1e-15)
                du_loss_between[bi] = (ws * (-torch.log(pb))).sum() / wsum
                du_loss_refl[bi] = (ws * (-torch.log(pr))).sum() / wsum

            du_loss = (1.0 - refl_du_weight) * du_loss_between + refl_du_weight * du_loss_refl
            losses.append(dl_loss + unlabeled_weight * du_loss)

        return torch.cat(losses)

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int,
                          between_pos_mask: torch.Tensor = None):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            if between_pos_mask is None:
                numerator = between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            else:
                pos_mask = between_pos_mask[mask].float()
                numerator = (between_sim * pos_mask).sum(1)
                numerator = numerator.clamp_min(1e-15)

            losses.append(-torch.log(
                numerator
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0,
             between_pos_mask: torch.Tensor = None,
             corrected: bool = False,
             du_pos_mask: torch.Tensor = None,
             du_pos_weight: torch.Tensor = None,
             du_pos_csr=None,
             du_pos_csr_t=None,
             unlabeled_weight: float = 1.0,
             corrected_variant: str = 'ifl-gr',
             refl_du_weight: float = 0.3):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if corrected:
            if du_pos_csr is None:
                assert du_pos_mask is not None
                assert du_pos_weight is not None

            if batch_size == 0:
                if corrected_variant == 'ifl-gc':
                    l1 = self.corrected_semi_loss_iflgc(
                        h1,
                        h2,
                        du_pos_mask=du_pos_mask,
                        du_pos_weight=du_pos_weight,
                        unlabeled_weight=unlabeled_weight,
                        refl_du_weight=refl_du_weight)
                    l2 = self.corrected_semi_loss_iflgc(
                        h2,
                        h1,
                        du_pos_mask=du_pos_mask.t(),
                        du_pos_weight=du_pos_weight.t(),
                        unlabeled_weight=unlabeled_weight,
                        refl_du_weight=refl_du_weight)
                else:
                    l1 = self.corrected_semi_loss(
                        h1,
                        h2,
                        du_pos_mask=du_pos_mask,
                        du_pos_weight=du_pos_weight,
                        unlabeled_weight=unlabeled_weight)
                    l2 = self.corrected_semi_loss(
                        h2,
                        h1,
                        du_pos_mask=du_pos_mask.t(),
                        du_pos_weight=du_pos_weight.t(),
                        unlabeled_weight=unlabeled_weight)
            else:
                if du_pos_csr is not None:
                    if du_pos_csr_t is None:
                        du_pos_csr_t = du_pos_csr

                    row_ptr, col_idx, col_w = du_pos_csr
                    row_ptr_t, col_idx_t, col_w_t = du_pos_csr_t

                    if corrected_variant == 'ifl-gc':
                        l1 = self.batched_corrected_semi_loss_iflgc_sparse(
                            h1,
                            h2,
                            du_row_ptr=row_ptr,
                            du_col_idx=col_idx,
                            du_col_w=col_w,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight,
                            refl_du_weight=refl_du_weight)
                        l2 = self.batched_corrected_semi_loss_iflgc_sparse(
                            h2,
                            h1,
                            du_row_ptr=row_ptr_t,
                            du_col_idx=col_idx_t,
                            du_col_w=col_w_t,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight,
                            refl_du_weight=refl_du_weight)
                    else:
                        l1 = self.batched_corrected_semi_loss_sparse(
                            h1,
                            h2,
                            du_row_ptr=row_ptr,
                            du_col_idx=col_idx,
                            du_col_w=col_w,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight)
                        l2 = self.batched_corrected_semi_loss_sparse(
                            h2,
                            h1,
                            du_row_ptr=row_ptr_t,
                            du_col_idx=col_idx_t,
                            du_col_w=col_w_t,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight)
                else:
                    if corrected_variant == 'ifl-gc':
                        l1 = self.batched_corrected_semi_loss_iflgc(
                            h1,
                            h2,
                            du_pos_weight=du_pos_weight,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight,
                            refl_du_weight=refl_du_weight)
                        l2 = self.batched_corrected_semi_loss_iflgc(
                            h2,
                            h1,
                            du_pos_weight=du_pos_weight.t(),
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight,
                            refl_du_weight=refl_du_weight)
                    else:
                        l1 = self.batched_corrected_semi_loss(
                            h1,
                            h2,
                            du_pos_weight=du_pos_weight,
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight)
                        l2 = self.batched_corrected_semi_loss(
                            h2,
                            h1,
                            du_pos_weight=du_pos_weight.t(),
                            batch_size=batch_size,
                            unlabeled_weight=unlabeled_weight)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
            return ret

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, between_pos_mask=between_pos_mask)
            l2 = self.semi_loss(
                h2,
                h1,
                between_pos_mask=None if between_pos_mask is None else between_pos_mask.t())
        else:
            l1 = self.batched_semi_loss(
                h1,
                h2,
                batch_size,
                between_pos_mask=between_pos_mask)
            l2 = self.batched_semi_loss(
                h2,
                h1,
                batch_size,
                between_pos_mask=None if between_pos_mask is None else between_pos_mask.t())

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
