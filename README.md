# Computer Vision Reliability Evaluation

Standard ML metrics such as accuracy, F1, and AUC measure how often a model is wrong, but they do not characterize *how* or *why* it fails. Knowing the how and why of failures is critical for aerospace systems. This framework reframes misclassification as a **failure event** to produce safety-relevant metrics that accuracy alone cannot provide specifically for computer vision.

The framework maps directly onto the five established components of reliability:
- Probability: how often the model succeeds,
- Durability: does accuracy survive distribution shift,
- Dependability: how confident the model is when it fails,
- Quality over Time: what is the failure mechanism,
- Availability: can the model perform under realistic operating conditions,

and combines them into a single composite score.

## Failure Event Definition

A failure event occurs when the predicted class does not match the true class:

$$P(F \mid D) = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}[\hat{y}_n \neq y_n] \cdot w(\tau_n)$$

*Editor note: Weights in aerospace systems should prioritize Type II errors, thus making this metric slightly different than plain accuracy.*

## Dataset Conditions

| Condition | Notation | Construction | Purpose |
|---|---|---|---|
| Baseline | $D$ | Clean test set | Ideal operating conditions |
| Durability | $D'$ | FFT low-frequency reconstruction | Feature-space distribution shift |
| Availability | $\tilde{D}$ | Stochastic pixel-space degradation | Real operating conditions |

**$D'$ and $\tilde{D}$ are deliberately distinct.** $D'$ shifts the deep feature distribution the model learned from by suppressing high-frequency texture components. $\tilde{D}$ applies pixel-space signal degradation (Gaussian noise, motion blur, brightness reduction, rotation) that preserves frequency statistics.

## Submetric Definitions

Probability $=1 - P(F \mid D)$

Durability $= 1-\frac{P(F \mid D') - P(F \mid D)}{P(F \mid D)}$ (normalized)

Dependability $=1 - \frac{\text{CVaR}_{0.95}(\mathcal{F}^*)}{2} + 0.5$ *(Editors Note: want to look into this more)*

Quality $= \exp(-\lambda \cdot \max(\beta_w - 1,\ 0)) \cdot \exp(-\gamma \cdot \max(1 - \beta_w,\ 0))$ (fit overconfident failures) *(Editors Note: not sure about this one yet, needs more work)*

Availability $= 1 - P(F \mid \tilde{D})$

## Results 

AlexNet on CIFAR-100

| Sub-Metric | Score | 95% CI |
|---|---|---|
| Probability Score | 0.5960 | [0.5856, 0.6054] |
| Durability Score | 0.9480 | [0.9349, 0.9600] |
| Dependability Score | 0.3975 | [0.3951, 0.3998] |
| Quality Score | 0.3490 | [0.3449, 0.6969] |
| Availability Score | 0.5051 | [0.4949, 0.5142] |
| **MRS** | **0.4954** | [0.4935, 0.5792] |

VGGNet on CIFAR-100

| Sub-Metric | Score | 95% CI |
|---|---|---|
| Probability Score | 0.7002 | [0.6912, 0.7084] |
| Durability Score | 0.4163 | [0.3761, 0.4554] |
| Dependability Score | 0.4082 | [0.4061, 0.4105] |
| Quality Score | 0.7061 | [0.3359, 0.9092] |
| Availability Score | 0.5913 | [0.5823, 0.6008] |
| **MRS** | **0.5326** | [0.4636, 0.5543] |

ResNet on CIFAR-100

| Sub-Metric | Score | 95% CI |
|---|---|---|
| Probability Score | 0.7377 | [0.7294, 0.7458] |
| Durability Score | 0.6401 | [0.6044, 0.6758] |
| Dependability Score | 0.4297 | [0.4278, 0.4315] |
| Quality Score | 0.7420 | [0.3416, 0.8426] |
| Availability Score | 0.6255 | [0.6158, 0.6349] |
| **MRS** | **0.6104** | [0.5078, 0.6227] |

## Future Implementations

- More statistical analysis of metrics
- Sensitivity analysis of metrics
- More dataset testing
- Diagnostic analysis
