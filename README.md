# Bayes-TrEx

Post-hoc explanation methods are gaining popularity as tools for interpreting, understanding, and debugging neural networks. Most post-hoc methods explain decisions in response to individual inputs. These individual inputs are typically drawn from the test set; however, the test set may be biased or may only sparsely invoke some model behaviours. To address these challenges, we introduce Bayes-TrEx, a model-agnostic method for generating distribution-conforming examples of known prediction confidence.  Using a classifier prediction and a data generator, Bayes-TrEx can be used to visualize class boundaries; to find in-distribution adversarial examples; to understand novel-class extrapolation; and to expose neural network overconfidence. We demonstrate Bayes-TrEx with rendered data (CLEVR) and organic data (MNIST, Fashion-MNIST).

<p align="center">
  <img src="./Images/level_set_overview.svg" alt="A visual overview of the main ideas behind Bayes-TrEx, showing a decision surface for a Corgi/Bread classifier and the associated level set slices.">

  Left: given a Corgi/Bread classifier, we generate *prediction level sets*, or sets of examples of a target prediction confidence. One way of finding such examples is by perturbing an arbitrary image to the target confidence (e.g., **p**<sub>Corgi</sub>=**p**<sub>Bread</sub>=0.5), as shown in (A). However, such examples give little insights into the typical model behavior because they are extremely unlikely in realistic situations.
  Bayes-TrEx explicitly considers a data distribution (gray shade on the rightmost plots) and finds in-distribution examples in a particular level set (e.g., likely Corgi (B), likely Bread (D), or ambiguous between Corgi and Bread (C)).

  Top Right: the classifier level set of **p**<sub>Corgi</sub>=**p**<sub>Bread</sub>=0.5 overlaid on the data distribution. Example (A) is unlikely to be sampled by Bayes-TrEx due to near-zero density under the distribution, while example (C) would be.

  Bottom Right: Sampling directly from the true posterior is infeasible, so we relax the formulation by "widening" the level set. By using different data distributions and confidences, Bayes-TrEx can uncover examples that invoke various model behaviors to improve model transparency.
</p>



## CLEVR Experiments

Instructions: `CLEVR/README.md`

## (Fashion-)MNIST Experiments

Instructions: `MNIST/README.md`
