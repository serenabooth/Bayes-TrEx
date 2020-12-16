# Bayes-TrEx

Post-hoc explanation methods are gaining popularity for interpreting, understanding, and debugging neural networks. Most analyses using such methods explain decisions in response to inputs drawn from the test set. However, the test set may have few examples that trigger some model behaviors, such as high-confidence failures or ambiguous classifications. To address these challenges, we introduce a flexible model inspection framework: Bayes-TrEx. Given a data distribution, Bayes-TrEx finds in-distribution examples with a specified prediction confidence. We demonstrate several use cases of Bayes-TrEx, including revealing highly confident (mis)classifications, visualizing class boundaries via ambiguous examples, understanding novel-class extrapolation behavior, and exposing neural network overconfidence. We use Bayes-TrEx to study classifiers trained on CLEVR, MNIST, and Fashion-MNIST, and we show that this framework enables more flexible holistic model analysis than just inspecting the test set. Code is online [here](https://github.com/serenabooth/Bayes-TrEx).

## CLEVR Experiments

Instructions: `CLEVR/README.md`

## (Fashion-)MNIST Experiments

Instructions: `MNIST/README.md`
