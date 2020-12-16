# Bayes-TrEx

Post-hoc explanation methods are gaining popularity as tools for interpreting, understanding, and debugging neural networks. Most post-hoc methods explain decisions in response to individual inputs. These individual inputs are typically drawn from the test set; however, the test set may be biased or may only sparsely invoke some model behaviours. To address these challenges, we introduce Bayes-TrEx, a model-agnostic method for generating distribution-conforming examples of known prediction confidence.  Using a classifier prediction and a data generator, Bayes-TrEx can be used to visualize class boundaries; to find in-distribution adversarial examples; to understand novel-class extrapolation; and to expose neural network overconfidence. We demonstrate Bayes-TrEx with rendered data (CLEVR) and organic data (MNIST, Fashion-MNIST).

<img align="center" src="./Images/level_set_overview.svg" alt="A visual overview of the main ideas behind Bayes-TrEx, showing a decision surface for a Corgi/Bread classifier and the associated level set slices.">



## CLEVR Experiments

Instructions: `CLEVR/README.md`

## (Fashion-)MNIST Experiments

Instructions: `MNIST/README.md`
