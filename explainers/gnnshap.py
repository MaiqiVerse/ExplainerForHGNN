import os
import math
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch import Tensor
from scipy.special import comb
from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanation, NodeExplanationCombination
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .node_dataset_scores import node_dataset_scores


# CUDA-BASED GNNShapSampler
_CUDA_GNNSHAP_EXT = None

def _load_gnnshap_cuda_extension(
    cuda_home: str | None = None,
    torch_cuda_arch_list: str | None = None,
    verbose: bool = False,
):
    """

    - Sets CUDA-related environment variables BEFORE importing/compiling:
        * CUDA_HOME
        * PATH += CUDA_HOME/bin
        * TORCH_CUDA_ARCH_LIST
    """
    global _CUDA_GNNSHAP_EXT
    if _CUDA_GNNSHAP_EXT is not None:
        return _CUDA_GNNSHAP_EXT

    if cuda_home is not None:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.path.join(
            cuda_home, "bin"
        )

    if torch_cuda_arch_list is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list

    from torch.utils.cpp_extension import load as load_cpp_extension

    _THIS_DIR = os.path.dirname(__file__)
    cu_file = os.path.join(_THIS_DIR, "cppextension", "cudagnnshap.cu")

    if not os.path.exists(cu_file):
        raise FileNotFoundError(f"CUDA file not found at: {cu_file}")

    # --- Load the CUDA extension ---
    _CUDA_GNNSHAP_EXT = load_cpp_extension(
        name="cudaGNNShapSampler",
        sources=[cu_file],
        extra_cflags=["-O2"],
        verbose=verbose,
    )
    return _CUDA_GNNSHAP_EXT


class CUDAGNNShapSampler:
    #CUDA implementation of GNNShap sampler.

    def __init__(self, nplayers: int, nsamples: int, **kwargs) -> None:
        if not isinstance(nplayers, int):
            raise TypeError("nplayers must be an integer.")
        assert nplayers > 1, "Number of players should be greater than 1"

        if not isinstance(nsamples, int):
            raise TypeError("nsamples must be an integer.")
        assert nsamples > 1, "Number of samples should be a positive number"

        self.nplayers = nplayers
        self.nsamples = nsamples

        # max_samples rule
        self.max_samples = 2 ** 30
        if self.nplayers <= 30:
            self.max_samples = 2 ** self.nplayers - 2

        # don't use more samples than max_samples
        self.nsamples = min(self.nsamples, self.max_samples)

        # CUDA kernel launch params
        self.num_blocks = kwargs.get("num_blocks", 16)
        self.num_threads = kwargs.get("num_threads", 128)

        # CUDA env/config
        cuda_home = kwargs.get("cuda_home", None)
        torch_cuda_arch_list = kwargs.get("torch_cuda_arch_list", None)
        verbose = kwargs.get("verbose", False)

        # Load the CUDA extension once and keep a handle
        self.cppsamp = _load_gnnshap_cuda_extension(
            cuda_home=cuda_home,
            torch_cuda_arch_list=torch_cuda_arch_list,
            verbose=verbose,
        )

    def sample(self) -> Tuple[Tensor, Tensor]:
        """Generate coalition mask_matrix and kernel_weights on CUDA.

        Returns:
            mask_matrix: (nsamples, nplayers) bool CUDA tensor
            kernel_weights: (nsamples,) float64 CUDA tensor
        """
        device = "cuda"

        mask_matrix = torch.zeros(
            (self.nsamples, self.nplayers),
            dtype=torch.bool,
            requires_grad=False,
            device=device,
        )
        kernel_weights = torch.zeros(
            (self.nsamples,),
            dtype=torch.float64,
            requires_grad=False,
            device=device,
        )

        # Call the CUDA kernel
        self.cppsamp.sample(
            mask_matrix,
            kernel_weights,
            self.nplayers,
            self.nsamples,
            self.num_blocks,
            self.num_threads,
        )

        return mask_matrix, kernel_weights


# CPU-BASED GNNShapSampler
class _CPUGNNSamplerCore:
    """
    CPU / PyTorch implementation of the GNNShap sampler
    that closely mimics the CUDA host logic.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def binomial_coef(self, n: int, k: int) -> int:
        if k > n or k < 0:
            return 0
        # exact combinatorial count
        return int(comb(n, k, exact=True))

    def generate_rth_combination(self, r: int, n: int, k: int) -> torch.Tensor:
        """
        Python/PyTorch version of rthComb:
        returns a boolean mask of length n corresponding
        to the r-th combination in lexicographic order.
        """
        mask = torch.zeros(n, dtype=torch.bool)
        x = 1

        for i in range(1, k + 1):
            y = self.binomial_coef(n - x, k - i)
            while y <= r:
                r = r - y
                x = x + 1
                if x > n:
                    return mask
                y = self.binomial_coef(n - x, k - i)
            mask[x - 1] = True
            x = x + 1

        return mask

    def sample(self, n_players: int, n_samples: int):
        """
        Main sampling function that generates coalition masks and kernel weights.

        This closely mirrors the CUDA host function `cudaSample`:
        - builds size_lookup and kernel_weights on the host
        - generates first-half masks (enumerated + random)
        - second half is the complement.
        """
        if n_samples % 2 != 0:
            raise ValueError("n_samples must be even for symmetric sampling")

        # OPTIONAL: make CPU sampler deterministic
        # torch.manual_seed(1234)

        n_half_samp = n_samples // 2

        # -------------------------------
        # 1) Host-side: size_lookup, kernel_weights, start_inds
        # -------------------------------
        size_lookup = torch.zeros(n_half_samp, dtype=torch.long)  
        kernel_weights = torch.zeros(n_samples, dtype=torch.float64) 

        n_subset_sizes = math.ceil((n_players - 1) / 2.0)
        n_paired_subset_sizes = math.floor((n_players - 1) / 2.0)

        coal_size_n_samples = [0] * n_subset_sizes
        start_inds = [0] * (n_subset_sizes + 1)

        # weight vector
        weight_vect = []
        for i in range(1, n_subset_sizes + 1):
            w = (n_players - 1.0) / (i * (n_players - i))
            weight_vect.append(w)

        # adjust middle coalition weight if unpaired
        if n_subset_sizes != n_paired_subset_sizes:
            weight_vect[n_paired_subset_sizes] /= 2.0

        weight_sum = sum(weight_vect)
        weight_vect = [w / weight_sum for w in weight_vect]
        rem_weight_vect = weight_vect.copy()

        n_samples_left = n_half_samp
        sum_kw = 0.0
        start_inds[0] = 0

        kw_write_pos = 0
        sl_write_pos = 0
        n_full_subsets = 0

        # ---- full enumeration sizes ----
        for i in range(1, n_subset_sizes + 1):
            n_subsets = self.binomial_coef(n_players, i)

            # middle coalition (unpaired) has half the subsets
            if i > n_paired_subset_sizes:
                if n_subsets % 2 != 0:
                    pass
                n_subsets //= 2

            if n_samples_left * rem_weight_vect[i - 1] + 1e-8 >= n_subsets:
                n_full_subsets += 1
                coal_size_n_samples[i - 1] = n_subsets
                n_samples_left -= n_subsets
                start_inds[i] = start_inds[i - 1] + n_subsets

                sum_kw += 50.0 * weight_vect[i - 1]
                # per-sample weight for these enumerated subsets
                per_kw = (50.0 * weight_vect[i - 1]) / n_subsets
                kernel_weights[kw_write_pos:kw_write_pos + n_subsets] = per_kw
                kw_write_pos += n_subsets

                size_lookup[sl_write_pos:sl_write_pos + n_subsets] = i
                sl_write_pos += n_subsets

                # rescale remaining weights (same as divideArray)
                if rem_weight_vect[i - 1] < 1.0:
                    scale = 1.0 - rem_weight_vect[i - 1]
                    for j in range(i - 1, n_subset_sizes):
                        rem_weight_vect[j] /= scale
            else:
                break

        # ---- remaining kernel weights for random region ----
        if n_samples_left > 0:
            rem_kw = (50.0 - sum_kw) / n_samples_left
        else:
            rem_kw = 0.0

        kernel_weights[kw_write_pos:kw_write_pos + n_samples_left] = rem_kw
        kw_write_pos += n_samples_left

        rnd_start_ind = n_half_samp - n_samples_left

        # ---- distribute remaining samples across subset sizes  ----
        if n_full_subsets != n_subset_sizes:
            rem_samples = n_samples_left
            round_up = True

            for i in range(n_full_subsets, n_subset_sizes - 1):
                if n_samples_left <= 0:
                    n_samples_left = 0
                    break

                if round_up:
                    count_i = min(
                        math.ceil(rem_samples * rem_weight_vect[i]),
                        n_samples_left,
                    )
                else:
                    count_i = min(
                        math.floor(rem_samples * rem_weight_vect[i]),
                        n_samples_left,
                    )

                coal_size_n_samples[i] = count_i
                n_samples_left -= count_i

                size_lookup[sl_write_pos:sl_write_pos + count_i] = i + 1
                sl_write_pos += count_i

                start_inds[i + 1] = start_inds[i] + count_i
                round_up = not round_up

            # remaining samples go to the last subset size
            coal_size_n_samples[n_subset_sizes - 1] = n_samples_left
            if n_samples_left > 0:
                size_lookup[sl_write_pos:sl_write_pos + n_samples_left] = n_subset_sizes
                sl_write_pos += n_samples_left
            n_samples_left = 0

        # mirror kernel weights to second half
        kernel_weights[n_half_samp:] = kernel_weights[:n_half_samp]

        # -------------------------------
        # 2) Build masks for first half
        # -------------------------------
        mask_first = torch.zeros(
            (n_half_samp, n_players),
            dtype=torch.bool,
        )

        # Full enumeration region
        for k in range(1, n_subset_sizes + 1):
            start = start_inds[k - 1]
            end = start_inds[k]
            n_samp_k = end - start
            if n_samp_k <= 0:
                continue

            # r runs from 0 to n_samp_k-1
            for local_r in range(n_samp_k):
                global_idx = start + local_r
                mask_first[global_idx] = self.generate_rth_combination(
                    r=local_r,
                    n=n_players,
                    k=k,
                )

        # Random region: i from rnd_start_ind to n_half_samp-1
        # If size_lookup[i] == 0 -> keep all zeros (empty coalition)
        for i in range(rnd_start_ind, n_half_samp):
            k = int(size_lookup[i].item())
            if k <= 0:
                continue  # empty coalition, leave row as zeros
            # random permutation of players
            perm = torch.randperm(n_players)
            chosen = perm[:k]
            mask_first[i, chosen] = True

        # -------------------------------
        # 3) Build full matrix with symmetry
        # -------------------------------
        mask_mat = torch.zeros(
            (n_samples, n_players),
            dtype=torch.bool,
            device=self.device,
        )

        mask_first = mask_first.to(self.device)
        mask_mat[:n_half_samp] = mask_first
        mask_mat[n_half_samp:] = ~mask_first  # symmetric complement

        kw_tensor = kernel_weights.to(self.device)

        return mask_mat, kw_tensor



class CPUGNNShapSampler:
    """
    Wrapper to match the CUDA sampler API:
    - __init__(nplayers, nsamples, ...)
    - sample() -> (mask_matrix, kernel_weights)
    """

    def __init__(
        self,
        nplayers: int,
        nsamples: int,
        num_blocks: int = 16,
        num_threads: int = 128,
        device: str = "cuda",
        **kwargs,
    ):
        assert isinstance(nplayers, int) and nplayers > 1, \
            "nplayers must be an integer > 1"
        assert isinstance(nsamples, int) and nsamples > 1, \
            "nsamples must be an integer > 1"

        self.nplayers = nplayers
        self.num_blocks = num_blocks  # kept for compatibility, unused
        self.num_threads = num_threads  # kept for compatibility, unused

        # Compute and store max_samples
        self.max_samples = 2 ** 30
        if self.nplayers <= 30:
            self.max_samples = 2 ** self.nplayers - 2

        # Cap nsamples by max_samples
        self.nsamples = min(nsamples, self.max_samples)

        # Ensure even number of samples for symmetric sampling
        if self.nsamples % 2 != 0:
            self.nsamples += 1

        # Initialize the actual sampler core
        self.sampler = _CPUGNNSamplerCore(device=device)

    def sample(self):
        """Generate samples and kernel weights using PyTorch implementation."""
        return self.sampler.sample(self.nplayers, self.nsamples)




### Solver 

# Base Solver
class BaseSolver(ABC):
    """Base class for Shapley solvers."""

    def __init__(
        self,
        mask_matrix: Tensor,
        kernel_weights: Tensor,
        yhat: Tensor,
        fnull: float,
        ffull: float,
        **kwargs
    ):
        self.mask_matrix = mask_matrix
        self.kernel_weights = kernel_weights
        self.yhat = yhat
        self.fnull = fnull
        self.ffull = ffull
        self.device = kwargs.get("device", "cpu")

    @abstractmethod
    def solve(self) -> np.ndarray:
        ...


# ================================================================
#   Weighted Least Squares Solver (WLS)
# ================================================================
class WLSSolver(BaseSolver):
    """Standard WLS Shapley solver (small / medium nplayers)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask_matrix = self.mask_matrix.to(self.device).double()
        self.kernel_weights = self.kernel_weights.to(self.device)
        self.yhat = self.yhat.to(self.device)

    def solve(self) -> np.ndarray:
        eyAdj = self.yhat - self.fnull
        eyAdj2 = eyAdj - self.mask_matrix[:, -1] * (self.ffull - self.fnull)

        etmp = self.mask_matrix[:, :-1] - self.mask_matrix[:, -1].unsqueeze(1)

        tmpT = (etmp * self.kernel_weights.unsqueeze(1)).transpose(0, 1)
        etmp_dot = torch.mm(tmpT, etmp)

        try:
            inv_mat = torch.linalg.inv(etmp_dot)
        except torch.linalg.LinAlgError:
            print("[WLS] Singular matrix — using pseudo-inverse.")
            inv_mat = torch.linalg.pinv(etmp_dot)

        w = torch.mm(inv_mat, torch.mm(tmpT, eyAdj2.unsqueeze(1)))[:, 0].cpu()

        phi = torch.zeros(self.mask_matrix.size(1))
        phi[:-1] = w
        phi[-1] = (self.ffull - self.fnull) - torch.sum(w)

        return phi.detach().numpy()


# ================================================================
#   Weighted Least Squares + Ridge (WLR)
# ================================================================
class WLRSolver(BaseSolver):
    """Regularized solver for large nplayers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ridge = kwargs.get("ridge", 1e-3)

        self.mask_matrix = self.mask_matrix.to(self.device).double()
        self.kernel_weights = self.kernel_weights.to(self.device)
        self.yhat = self.yhat.to(self.device)

    def solve(self) -> np.ndarray:
        eyAdj = self.yhat - self.fnull
        eyAdj2 = eyAdj - self.mask_matrix[:, -1] * (self.ffull - self.fnull)

        etmp = self.mask_matrix[:, :-1] - self.mask_matrix[:, -1].unsqueeze(1)

        tmpT = (etmp * self.kernel_weights.unsqueeze(1)).transpose(0, 1)
        etmp_dot = torch.mm(tmpT, etmp)

        etmp_dot = etmp_dot + self.ridge * torch.eye(
            etmp_dot.shape[0], device=self.device
        )

        try:
            inv_mat = torch.linalg.inv(etmp_dot)
        except torch.linalg.LinAlgError:
            print("[WLR] Singular — using pseudo-inverse.")
            inv_mat = torch.linalg.pinv(etmp_dot)

        w = torch.mm(inv_mat, torch.mm(tmpT, eyAdj2.unsqueeze(1)))[:, 0].cpu()

        phi = torch.zeros(self.mask_matrix.size(1))
        phi[:-1] = w
        phi[-1] = (self.ffull - self.fnull) - torch.sum(w)

        return phi.detach().numpy()


# ================================================================
#   Solver Factory
# ================================================================
def get_solver(
    solver_name: str,
    mask_matrix: Tensor,
    kernel_weights: Tensor,
    yhat: Tensor,
    fnull: float,
    ffull: float,
    **kwargs
) -> BaseSolver:

    solvers = {
        "WLSSolver": WLSSolver,
        "WLRSolver": WLRSolver,
    }

    if solver_name not in solvers:
        raise KeyError(
            f"Solver '{solver_name}' not available. Choose: {list(solvers.keys())}"
        )

    return solvers[solver_name](
        mask_matrix,
        kernel_weights,
        yhat,
        fnull,
        ffull,
        **kwargs,
    )




class GNNShapCore(ExplainerCore):

    def __init__(self, config):
        super().__init__(config)

        self.nsamples = config.get("nsamples", 15000)
        self.n_hop = config.get("n_hop", 2)

        # Sampler type: "cuda_based" (default) or "cpu"
        # Also accept mis-typed key "samper_type" for robustness.
        self.sampler_type = config.get(
            "sampler_type",
            config.get("samper_type", "cuda_based"),
        )

        # CUDA-related config (used only if sampler_type is CUDA)
        self.cuda_home = config.get("cuda_home", None)
        self.torch_cuda_arch_list = config.get("torch_cuda_arch_list", None)
        self.cuda_num_blocks = config.get("cuda_num_blocks", 16)
        self.cuda_num_threads = config.get("cuda_num_threads", 128)
        self.cuda_verbose = config.get("cuda_verbose", False)

    # ==================================================================
    def _build_sampler(self, nplayers: int):
        """
        Factory for sampler instances based on self.sampler_type.

        Supports:
        - "cuda_based" / "cuda" / "gpu"  -> CUDAGNNShapSampler
        - "cpu" / "cpu_based"           -> CPUGNNShapSampler
        """
        st = str(self.sampler_type).lower()

        if st in ("cuda_based", "cuda", "gpu"):
            return CUDAGNNShapSampler(
                nplayers=nplayers,
                nsamples=self.nsamples,
                cuda_home=self.cuda_home,
                torch_cuda_arch_list=self.torch_cuda_arch_list,
                num_blocks=self.cuda_num_blocks,
                num_threads=self.cuda_num_threads,
                verbose=self.cuda_verbose,
            )

        elif st in ("cpu", "cpu_based"):
            # Pure PyTorch / CPU sampler (no custom CUDA extension)
            return CPUGNNShapSampler(
                nplayers=nplayers,
                nsamples=self.nsamples,
                num_blocks=self.cuda_num_blocks,
                num_threads=self.cuda_num_threads,
                device=self.device_string,
            )

        else:
            raise ValueError(
                f"Unknown sampler_type='{self.sampler_type}'. "
                "Expected one of: 'cuda_based', 'cuda', 'gpu', 'cpu', 'cpu_based'."
            )

    @torch.no_grad()
    def explain(self, model, **kwargs):
        """
        pipeline:
        - Extract neighbors
        - Compute f_full
        - Compute φ for each meta-path
        - Build top-k hard masks
        - Build final explanation
        """
        self.model = model
        self.model.eval()

        self.node_id = kwargs.get("node_id")
        if self.node_id is None:
            raise ValueError("node_id is required")
        
        print(f"[GNNShapCore] >>> Starting explanation for node_id = {self.node_id}")

        # ---------------------------------------------------------------
        # 1) Extract local subgraphs (per meta-path)
        # ---------------------------------------------------------------
        gs, features = self.extract_neighbors_input()
        self.gs = [g.to(self.device_string) for g in gs]
        self.features = features.to(self.device_string)
        self.target_local = self.recovery_dict[self.node_id]

        # ---------------------------------------------------------------
        # 2) Compute f_full
        # ---------------------------------------------------------------
        self.ffull, self.target_class = self._compute_ffull(self.gs, self.features)

        # ---------------------------------------------------------------
        # 3) Compute φ for each meta-path
        # ---------------------------------------------------------------
        phi_list = []
        for mp_idx in range(len(self.gs)):
            phi_list.append(self._compute_phi_for_metapath(mp_idx))


        self.edge_mask = phi_list

        # Identity feature mask (we do not mask features)
        self.feature_mask = [
            torch.ones(self.features.shape[1], device=self.device_string)
        ]

        return self.construct_explanation()
    


    # ==================================================================
    # Compute φ for one meta-path
    # ==================================================================
    def _compute_phi_for_metapath(self, mp_idx):
        g = self.gs[mp_idx].coalesce()
        E = g.indices().size(1)


        if E <= 1:
            # One or zero edges → Shapley not defined → return zero vector
            return torch.zeros(E, device=self.device_string)        

        sampler = self._build_sampler(E)


    
        mask_matrix, kernel_weights = sampler.sample()


        fnull = self._compute_fnull(mp_idx)
        yhat = self._compute_yhat(mp_idx, mask_matrix)



        solver_name = "WLRSolver" if E > 1000 else "WLSSolver"
        solver = get_solver(
            solver_name=solver_name,
            mask_matrix=mask_matrix,
            kernel_weights=kernel_weights,
            yhat=yhat,
            fnull=fnull,
            ffull=self.ffull,
            device=self.device_string,
        )

        phi = solver.solve()
        

        return torch.tensor(phi, device=self.device_string)

    # ==================================================================
    # f_full
    # ==================================================================
    def _compute_ffull(self, gs, features):
        logits = self.model.custom_forward(lambda _: (gs, features))
        node_logits = logits[self.target_local]
        cls = node_logits.argmax().item()
        return node_logits[cls].item(), cls

    # ==================================================================
    # f_null (meta-path removed)
    # ==================================================================
    def _compute_fnull(self, mp_idx):
        empty_gs = []
        for i, g in enumerate(self.gs):
            if i == mp_idx:
                idx = torch.empty((2, 0), dtype=torch.long, device=self.device_string)
                val = torch.empty(0, dtype=torch.float, device=self.device_string)
                eg = torch.sparse_coo_tensor(idx, val, size=g.size(),
                                             device=self.device_string)
                empty_gs.append(eg)
            else:
                empty_gs.append(g)

        logits = self.model.custom_forward(lambda _: (empty_gs, self.features))
        nl = logits[self.target_local]
        return nl[nl.argmax()].item()

    # ==================================================================
    # yhat (prediction for each coalition)
    # ==================================================================
    def _compute_yhat(self, mp_idx, mask_matrix):
        g = self.gs[mp_idx].coalesce()
        idxs, vals = g.indices(), g.values()

        outputs = []
        for mask in mask_matrix:
            masked_idxs = idxs[:, mask]
            masked_vals = vals[mask]

            masked_g = torch.sparse_coo_tensor(
                masked_idxs, masked_vals, size=g.size(), device=self.device_string
            ).coalesce()

            masked_gs = [
                masked_g if j == mp_idx else self.gs[j]
                for j in range(len(self.gs))
            ]

            logits = self.model.custom_forward(lambda _: (masked_gs, self.features))
            nl = logits[self.target_local]
            outputs.append(nl[nl.argmax()].item())

        return torch.tensor(outputs, device=self.device_string)

    

    # ==================================================================
    # Build final explanation (for fidelity)
    # ==================================================================
    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)

        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation


    @property
    def edge_mask_for_output(self):
        return [phi.clone().detach() for phi in self.edge_mask]

    @property
    def feature_mask_for_output(self):
        return None

    # Node id mapping (global → local in subgraph)
    def mapping_node_id(self):
        if getattr(self, "mapped_node_id", None) is not None:
            return self.mapped_node_id

        if not self.config.get("extract_neighbors", True):
            self.mapped_node_id = self.node_id
        else:
            self.mapped_node_id = self.recovery_dict[self.node_id]

        return self.mapped_node_id

    # Neighbor extraction (k-hop pruned subgraph)
    def extract_neighbors_input(self):
        """
        Extract the neighbors of the node to be explained
        :return:
        """
        if not self.config.get('extract_neighbors', True):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            return gs, features

        if getattr(self, 'neighbor_input',
                   None) is not None and self.neighbor_input.get(
            "gs", None) is not None:
            return self.neighbor_input["gs"], self.neighbor_input["features"]

        # we follow the default value in hencex
        self.n_hop = self.config.get('n_hop', 2)

        gs, features = self.model.standard_input()

        used_nodes_set = set()

        for g in gs:
            indices = g.indices()

            # consider memory-efficient
            current_nodes = [self.node_id]

            for i in range(self.n_hop):
                new_current_nodes = set()
                for node in current_nodes:
                    mask = (indices[0] == node) | (indices[1] == node)
                    used_nodes_set.update(indices[1][mask].tolist())
                    used_nodes_set.update(indices[0][mask].tolist())
                    new_current_nodes.update(indices[1][mask].tolist())
                    new_current_nodes.update(indices[0][mask].tolist())

                new_current_nodes = list(new_current_nodes)
                current_nodes = new_current_nodes

        self.used_nodes = sorted(list(used_nodes_set))
        self.recovery_dict = {node: i for i, node in enumerate(self.used_nodes)}
        self._quick_transfer = torch.zeros(len(features), dtype=torch.long
                                           ).to(self.device_string)
        for i, node in enumerate(self.used_nodes):
            self._quick_transfer[node] = i

        # now reconstruct the graph
        temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device_string)
        new_gs = []
        for g in gs:
            indices = g.indices()
            if self.n_hop == 1:
                # only keep the edges connected to the target node
                mask = (indices[0] == self.node_id) | (indices[1] == self.node_id)
            else:
                # we do not strict to only keep the edges in the paths between the target node and its neighbors, but keep all edges among
                # the used nodes. This is to avoid too much computation in finding the corresponding edges.
                mask = torch.isin(indices[0], temp_used_nodes_tensor) & \
                    torch.isin(indices[1], temp_used_nodes_tensor)
            # use self._quick_transfer to speed up
            new_indices = torch.stack(
                [self._quick_transfer[indices[0][mask]],
                 self._quick_transfer[indices[1][mask]]],
                dim=0)
            new_indices = new_indices.to(self.device_string)
            new_values = g.values()[mask]
            shape = torch.Size([len(self.used_nodes), len(self.used_nodes)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape))

        self.neighbor_input = {"gs": new_gs, "features": features[self.used_nodes]}
        return self.neighbor_input["gs"], self.neighbor_input["features"]


    #  Custom handle_fn for fidelity / other uses
    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Returns a handle_fn compatible with model.custom_forward(),
        used by fidelity preparation functions.
        """

        def handle_fn(model):
            gs, feats = self.extract_neighbors_input()

            if masked_gs is not None:
                gs = []
                for g in masked_gs:
                    mask = g.values() != 0
                    indices = g.indices()[:, mask]
                    values = g.values()[mask]
                    shape = g.shape
                    gs.append(torch.sparse_coo_tensor(indices, values, shape))
                gs = [i.to(self.device_string) for i in gs]

            return gs, feats

        return handle_fn


class GNNShap(Explainer):
    """
    Dataset wrapper: Runs GNNExplainerMetaCore on all test nodes
    and computes dataset-level fidelity metrics.
    """

    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model

        if not getattr(model.dataset, "single_graph", True):
            raise ValueError("Only single-graph datasets are supported.")

        return self._explain_dataset(**kwargs)

    def _explain_dataset(self, **kwargs):
        core_class = self.core_class()
        test_labels = self.model.dataset.labels[2]

        max_nodes = kwargs.get("max_nodes", None)
        if max_nodes is not None:
            test_labels = test_labels[:max_nodes]

        explanations = []
        for idx, _ in test_labels:
            core = core_class(self.config)
            core.to(self.device)
            explanations.append(core.explain(self.model, node_id=idx))

        combined = NodeExplanationCombination(node_explanations=explanations)

        if self.config.get("control_data", None) is not None:
            combined.control_data = self.config["control_data"]

        self._evaluate(combined)
        self._save_summary()

        return self.eval_result

    def _evaluate(self, combined):
        eval_result = {}

        if self.config.get("eval_metrics", None):
            for metric in self.config["eval_metrics"]:
                combined = prepare_combined_explanation_fn_for_node_dataset_scores[metric](
                    combined, self
                )
                eval_result[metric] = node_dataset_scores[metric](combined)

        self.eval_result = eval_result
        return eval_result

    def _save_summary(self):
        path = self.config.get("summary_path", None)
        if path is None:
            return
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.eval_result, f)

    def core_class(self):
        return GNNShapCore