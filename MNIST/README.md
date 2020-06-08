# MNIST and Fashion-MNIST

Before running any code, please download the pre-trained models at `https://drive.google.com/file/d/1ZrDJb4-CCX1_X6ulL5smjvXzrMuaFKsv/view?usp=sharing`, and extract it to replace the empty `saved_models/` folder. If done correctly, you should find `saved_models/mnist_gan.pth` and `saved_models/all_but_one/fashion_mnist_gan_wo_0.pth`, as two examples.

All codes are written in `Python 3`, with `pytorch`  and `pyro` as the deep learning and inference libraries. For MNIST and Fashion-MNIST, the code file name starts with `mnist_`. The usecases are detailed below. The sampled latent vectors will be written into the currently empty `results/` folder.

1. `python3 mnist_inspect.py --nn-model gan|vae --dataset mnist|fashion_mnist --type AvB|unif|hC`: sample in-distribution examples that satisfy a criterion. `--nn-model` can be `gan` or `vae`, `--dataset` can be `mnist` or `fashion_mnist`, and `--type` can be:
	1. `AvB`: to sample examples that are ambivalent between class `A` and `B`, e.g. `1v7`.
	2. `unif`: to sample examples that are uniformly ambivalent among all classes.
	3. `hC`: to sample high-confidence examples for class `C`, e.g. `h0` for high confident images of class `0`.
2. `python3 mnist_graded.py --nn-model gan|vae --dataset mnist|fashion_mnist --type AvB`: sample graded confidence interpolation between class `A` and `B`, e.g. `8v9`. All other parameters are same as above.
3. `python3 mnist_adversarial.py --nn-model gan|vae --dataset mnist|fashion_mnist --excl-label C`: sample in-distribution adversarial examples for class `C`.
4. `python3 mnist_extrapolation.py --nn-model gan|vae --dataset mnist|fashion_mnist --label C`: sample extrapolation examples for class `C`. For MNIST, `C` can be 0, 1, 3, 6, or 9, and for Fashion-MNIST, `C` can be 2, 3, 5, 6, or 9.
5. ``python3 mnist_adda.py --model baseline|adda --label C``: sample high-confidence examples for class `C` in the domain adaptation experiment. `--model` can be:
	1. `baseline`: use the baseline model (trained on SVHN).
	2. `adda`: use the domain-adaptation model (trained on SVHN and unlabeled MNIST).

Each of the above scripts writes a message to the console like:
`Sample: 100%|█| 2000/2000 [18:26,  1.81it/s, step size=5.94e-03, acc. prob=0.811
`
This message includes the sampled probability (e.g., 0.811). 
saves a .txt file and a batch of images to the `results/` directory.
