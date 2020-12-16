# Bayes-TrEx: A Bayesian Sampling Approach to Model Transparency by Example

Post-hoc explanation methods are gaining popularity for interpreting, understanding, and debugging neural networks. Most analyses using such methods explain decisions in response to inputs drawn from the test set. However, the test set may have few
examples that trigger some model behaviors, such as high-confidence failures or ambiguous classifications. To address these challenges, we introduce a flexible model inspection framework: Bayes-TrEx. Given a data distribution, Bayes-TrEx finds in-distribution examples with a specified prediction confidence. We demonstrate several use cases of Bayes-TrEx, including revealing highly confident (mis)classifications, visualizing class boundaries via ambiguous examples, understanding novel-class extrapolation behavior, and exposing neural network overconfidence. We use Bayes-TrEx to study classifiers trained on CLEVR, MNIST, and Fashion-MNIST, and we show that this framework enables more flexible holistic model analysis than just inspecting the test set. Code is available at [https://github.com/serenabooth/Bayes-TrEx](https://github.com/serenabooth/Bayes-TrEx).

## Method Overview: Analyzing a Corgi/Bread Classifier

<p align="center">
  <img src="./Images/level_set_overview.svg" alt="A visual overview of the main ideas behind Bayes-TrEx, showing a decision surface for a Corgi/Bread classifier and the associated level set slices.">

  Left: given a Corgi/Bread classifier, we generate *prediction level sets*, or sets of examples of a target prediction confidence. One way of finding such examples is by perturbing an arbitrary image to the target confidence (e.g., **p**<sub>Corgi</sub>=**p**<sub>Bread</sub>=0.5), as shown in (A). However, such examples give little insights into the typical model behavior because they are extremely unlikely in realistic situations.
  Bayes-TrEx explicitly considers a data distribution (gray shade on the rightmost plots) and finds in-distribution examples in a particular level set (e.g., likely Corgi (B), likely Bread (D), or ambiguous between Corgi and Bread (C)).

  Top Right: the classifier level set of **p**<sub>Corgi</sub>=**p**<sub>Bread</sub>=0.5 overlaid on the data distribution. Example (A) is unlikely to be sampled by Bayes-TrEx due to near-zero density under the distribution, while example (C) would be.

  Bottom Right: Sampling directly from the true posterior is infeasible, so we relax the formulation by "widening" the level set. By using different data distributions and confidences, Bayes-TrEx can uncover examples that invoke various model behaviors to improve model transparency.
</p>

## Paper (with Appendix)

The paper and appendix are online [here](./bayestrex_full_paper.pdf).

## Experiments

Bayes-TrEx requires a data distribution---whether manually-defined or learned. We demonstrate Bayes-TrEx on CLEVR, with manually-defined scene graphs. We also demonstrate Bayes-TrEx on Fashion-MNIST and MNIST, with learned VAEs or GANs to represent the data distribution.

[CLEVR Code](./CLEVR)

[(Fashion-)MNIST Code](./MNIST)

## Citation

@inproceedings{booth21:bayestrex,  
&emsp;  title = {Bayes-TrEx: A Bayesian Sampling Approach to
                 Model Transparency by Example}  
&emsp;  author = {Serena Booth and Yilun Zhou and Ankit Shah and Julie Shah},  
&emsp;  booktitle = {AAAI},  
&emsp;  year = {2021},  
}


## Questions/comments?

Email {serenabooth, yilun, ajshah, julie_a_shah}@csail.mit.edu
