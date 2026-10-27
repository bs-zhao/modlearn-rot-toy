"""Canonical mathematical metadata for every numbered functional metric.

A metric is specified once by a name, formula, numerical implementation,
interpretation, and expected behavior. Never reorder existing registrations.
"""
from dataclasses import dataclass
from collections import OrderedDict

@dataclass(frozen=True)
class MetricSpec:
    id: str
    key: str
    name: str
    formula: str
    numerical_method: str
    meaning: str
    behavior: str

METRICS=OrderedDict()
BY_KEY={}

def register_metric(key,name,formula,numerical_method,meaning,behavior="candidate"):
    if key in BY_KEY:raise ValueError(f"Duplicate metric key: {key}")
    identifier=f"M{len(METRICS)+1:03d}"
    spec=MetricSpec(identifier,key,name,formula,numerical_method,meaning,behavior)
    METRICS[identifier]=spec
    BY_KEY[key]=identifier
    return identifier

def metric_id(key):return BY_KEY[key]

register_metric("xu_group_l1", "RMS XU group sparsity", "\\sum_k e_k/(\\sqrt K\\|e\\|_2)", "Autograd XU; RMS across X/U; normalized L1", "Concentration of mixed XU effects across U axes", "candidate")
register_metric("xu_energy_concentration", "XU group energy diversity", "1-\\sum_k(e_k^2/\\sum e_j^2)^2", "XU gradient squares; Gini-Simpson energy", "Small if one axis carries most XU energy", "candidate")
register_metric("local_xu_l1", "Local XU row sparsity", "\\mathbb E_{\\|G_j\\|>0}\\|G_j\\|_1/\\|G_j\\|_2", "Pointwise normalized XU rows; zero rows excluded", "Local single-axis gradient dependence; T5/T9 counterexample", "candidate")
register_metric("grad_xcov_offdiag", "X-conditional U-gradient covariance JD", "\\|\\mathrm{off}(C)\\|_F^2/\\|C\\|_F^2,\\ C=\\mathbb E_U\\mathrm{Cov}_X(\\nabla_u h)", "Center within X at each U then integrate U", "How stimulus-dependent U gradients co-vary", "candidate")
register_metric("grad_gram_offdiag", "Uncentered U-gradient Gram JD", "\\|\\mathrm{off}(B)\\|_F^2/\\|B\\|_F^2,\\ B=\\mathbb E_{X,U}gg^\\top", "Uncentered gradient outer product over X/U", "Functional orthogonality of U-gradient directions", "candidate")
register_metric("uu_offdiag", "UU cross-curvature fraction", "\\mathbb E\\sum_{i\\neq j}H_{ij}^2/\\mathbb E\\|H\\|_F^2", "UU Hessian (autograd); energy ratio", "T3 45-degree mixing falsely improves it", "candidate")
register_metric("uu_diagonal", "UU self-curvature fraction", "\\mathbb E\\sum_iH_{ii}^2/\\mathbb E\\|H\\|_F^2", "Diagonal UU Hessian energy / total", "Complement of M006 when Hessian nonzero", "diagnostic")
register_metric("u_grad_energy", "Total U-gradient energy", "\\mathbb E\\|\\nabla_u h\\|_2^2", "Squared U gradient then average", "Orthogonal-rotation invariant control", "invariant")
register_metric("xu_total_energy", "Total XU mixed-gradient energy", "\\mathbb E\\|\\nabla_x\\nabla_u h\\|_F^2", "Squared mixed derivatives then average", "Orthogonal-rotation invariant control", "invariant")
register_metric("x_gradient_energy", "Total X-gradient energy", "\\mathbb E\\|\\nabla_x h\\|_2^2", "Squared X gradient then average", "Invariant under reparameterization at fixed physical points", "invariant")
register_metric("x_curvature_energy", "Total XX Hessian energy", "\\mathbb E\\|H_{XX}\\|_F^2", "Squared XX derivatives then average", "Invariant under U-coordinate rotation", "invariant")
register_metric("sweep_affine", "U-sweep affine approximation error", "\\frac1K\\sum_k\\min_{\\alpha_x,\\beta_x}\\mathrm{MSE}(S_k-\\alpha_x-\\beta_x s)/\\mathrm{Var}(S_k)", "Uniform U-axis sweep; least-squares line per X", "Detects U-axis non-affine dependence", "candidate")
register_metric("sweep_shape_min", "Minimum normalized response-shape variance", "\\min_k\\mathbb E_x\\mathrm{Var}_s[S_k(s,x)/\\|S_k(s,\\cdot)\\|_2]", "Normalize response vectors along X; variance over sweep", "Tests existence of at least one pure-amplitude axis", "diagnostic")
register_metric("sweep_rank1_min", "Minimum rank-one U-sweep residual", "\\min_k[1-\\sigma_1(S_k)^2/\\|S_k\\|_F^2]", "Sweep matrix eigen/SVD top singular value", "Existence of a near rank-one functional sweep", "diagnostic")
register_metric("sweep_curvature", "Finite-difference U sweep curvature", "\\frac1K\\sum_k\\mathbb E[(\\Delta_s^2S_k/\\Delta s^2)^2]", "Three-point centered sweep second difference", "Average U-axis nonlinearity", "candidate")
register_metric("rectangle_uu", "UU rectangular difference interaction", "\\mathbb E_{k<l,X}[(S_{++}-S_{+-}-S_{-+}+S_{--})/(4\\delta^2)]^2", "Four corners for each U-axis pair at fixed center", "Nonlocal UU interaction; fails T3 if minimized alone", "candidate")
register_metric("operator_v1_jd", "Function-operator v1 joint diagonalization", "\\mathbb E_{\\alpha\\in\\mathcal A_1}\\|\\mathrm{off}(RK_\\alpha R^\\top)\\|_F^2", "Level+first-X signatures; integrate X, outer product, integrate U, normalize", "Common axes of first-order functional sensitivity operators", "candidate")
register_metric("operator_v2_jd", "Function-operator v2 joint diagonalization", "\\mathbb E_{\\alpha\\in\\mathcal A_2}\\|\\mathrm{off}(RK_\\alpha R^\\top)\\|_F^2", "Level+first+diagonal/mixed second-X operators", "Common axes of first/second-order functional operators", "candidate")
register_metric("operator_v1_level", "Level-operator offdiagonal loss", "J_{\\{\\psi_0\\}}(R)", "v1 retained level operator normalized then JD", "Joint effect averaged over all X", "candidate")
register_metric("operator_v1_first", "First-X operator offdiagonal loss", "J_{\\{\\psi_j\\}}(R)", "v1 retained first X derivative operators", "Dependence of U sensitivity on X coordinates", "candidate")
register_metric("operator_v2_second_diagonal", "Diagonal second-X operator JD", "J_{\\{\\psi_{jj}\\}}(R)", "v2 retained diagonal second-X operators", "Second-order single-X effect directions", "candidate")
register_metric("operator_v2_second_mixed", "Mixed second-X operator JD", "J_{\\{\\psi_{jl}:j<l\\}}(R)", "v2 retained mixed XX operators", "Across-X interaction effect directions", "candidate")
register_metric("operator_fisher_offdiag", "Sigmoid-weighted gradient Gram offdiagonal", "\\|\\mathrm{off}(RI_wR^\\top)\\|_F^2", "Iw=E[sigmoid(h)(1-sigmoid(h))gg^T], unnormalized", "Functional weighted information; NOT likelihood Fisher for continuous Y", "candidate")
register_metric("operator_fisher_trace", "Sigmoid-weighted gradient Gram trace", "\\mathrm{tr}(RI_wR^\\top)", "Trace of the same weighted Gram matrix", "Invariant weighted total gradient magnitude", "invariant")
