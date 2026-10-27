"""T11: original CR3/CR2-4P gain-loss lottery value difference.

Exact CR3 functional structure (original project:
experiment/choice/research_based/s0_cr2_design_recovery.py):

    eta = tau*(V(A)-V(B))
    V(g,l,p) = w(p,gamma)*(g/100)**alpha
             - lambda*w(1-p,gamma)*(l/100)**alpha
    w(p,gamma) = sigmoid(gamma*logit(clip(p,1e-12,1-1e-12)))

Toy coordinates are *unit-cube* normalized, not original CR3 physical priors.
X=(gA,lA,pA,gB,lB,pB) contains gain/loss amounts normalized by 200 USD,
probabilities unchanged; theta=(alpha_u,lambda_u,gamma_u,tau_u).

Physical CR3 parameter bounds were alpha [.02,.999], lambda [.3,5],
gamma [.02,.999], tau [.5,4]. The toy maps each u in [0,1] affinely
to its corresponding physical interval. This is not original CR3's
500-item fixed stimulus design nor its beta/gamma/truncated-normal priors.

The base output eta is the CR3 logit. T11-id returns eta directly
(strips ONLY the last outer sigmoid from original CR3 P(A)=sigmoid(eta)).
T11-sig EXACTLY restores original CR3's deterministic P(A) function.
T11-sq/tanh/sp/exp each apply their transform to eta, NOT to sigmoid(eta).
The sigmoid INSIDE probability_weight(p,gamma) is always retained.
This continuous toy never samples a Bernoulli outcome.
"""
import torch
from ..models import register_model

# Original CR2-4P physical parameter bounds, used unchanged.
ALPHA_LO, ALPHA_HI = 0.02, 0.999
LAMBDA_LO, LAMBDA_HI = 0.3, 5.0
GAMMA_LO, GAMMA_HI = 0.02, 0.999
TAU_LO, TAU_HI = 0.5, 4.0
CR3_AMOUNT_SCALE_USD = 100.0
MAX_TOY_AMOUNT_USD = 200.0
WEIGHTING_CLIP = 1e-12

# On the physical input cube, 0 <= gain/100,loss/100 <= 2.
# Per lottery V is in [-5*2**.999, 2**.999]; the signed difference
# times tau<=4 has these guaranteed bounding limits (tight in the
# continuous weighting limit; probability endpoint clipping is tiny).
ETA_ABS_BOUND = TAU_HI * (1 + LAMBDA_HI) * (
    MAX_TOY_AMOUNT_USD / CR3_AMOUNT_SCALE_USD
) ** ALPHA_HI

def physical_parameters(theta):
    """Normalized toy coordinates -> physical alpha, loss lambda, gamma, tau."""
    a = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * theta[..., 0]
    lam = LAMBDA_LO + (LAMBDA_HI - LAMBDA_LO) * theta[..., 1]
    gam = GAMMA_LO + (GAMMA_HI - GAMMA_LO) * theta[..., 2]
    tau = TAU_LO + (TAU_HI - TAU_LO) * theta[..., 3]
    return a, lam, gam, tau

def probability_weight(p, gamma):
    """Original CR3 logit-domain weighting, with its 1e-12 clipping."""
    clipped = p.clamp(WEIGHTING_CLIP, 1.0 - WEIGHTING_CLIP)
    return torch.sigmoid(gamma * torch.logit(clipped))

def lottery_value(g, loss, p, alpha, lam, gamma):
    """g/loss are nonnegative USD amounts; loss is a magnitude, not signed."""
    wg = probability_weight(p, gamma)
    wl = probability_weight(1.0 - p, gamma)
    scaled_gain = (g / CR3_AMOUNT_SCALE_USD).pow(alpha)
    scaled_loss = (loss / CR3_AMOUNT_SCALE_USD).pow(alpha)
    return wg * scaled_gain - lam * wl * scaled_loss

@register_model(
    "T11", k=4, dx=6,
    description="CR3 four-parameter lottery logit: tau*(V(A)-V(B)); unit-cube inputs",
    eta_bounds=(-ETA_ABS_BOUND, ETA_ABS_BOUND),
)
def t11(x, theta):
    alpha, lam, gamma, tau = physical_parameters(theta)
    g_a, l_a, p_a, g_b, l_b, p_b = (x[..., i] for i in range(6))
    # Convert normalized toy amounts back to raw CR3 USD amounts.
    v_a = lottery_value(
        g_a * MAX_TOY_AMOUNT_USD, l_a * MAX_TOY_AMOUNT_USD,
        p_a, alpha, lam, gamma,
    )
    v_b = lottery_value(
        g_b * MAX_TOY_AMOUNT_USD, l_b * MAX_TOY_AMOUNT_USD,
        p_b, alpha, lam, gamma,
    )
    return tau * (v_a - v_b)
