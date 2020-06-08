
from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch import nn
from nets import *

__all__ = ['VAEModel', 'GANModel', 'ADDAModel']

class NNModel(object):
    '''an abstract base class that provide predict_from_latent() method'''
    def __init__(self):
        self.latent_dim = None

    def predict_from_latent(self, z, return_img=False):
        raise NotImplemented

class VAEModel(NNModel):
    def __init__(self, dset, latent_dim=5, vae_path=None, classifier_path=None, temperature=1., device='cuda'):
        super(VAEModel, self).__init__()
        assert dset in ['mnist', 'fashion_mnist'], 'this model only supports mnist fashion_mnist'
        self.latent_dim = latent_dim
        self.device = device

        if vae_path is None:
            print('using default VAE')
            vae_path = 'saved_models/%s_vae.pth'%dset
        VAE = {'mnist': MNISTVAE, 'fashion_mnist': FashionMNISTVAE}[dset]
        self.vae = VAE(latent_dim)
        self.vae.load_state_dict(torch.load(vae_path))
        self.vae.to(device)
        self.vae.eval()

        if classifier_path is None:
            print('using default classifier')
            classifier_path = 'saved_models/%s_classifier.pth'%dset
        Classifier = {'mnist': MNISTClassifier, 'fashion_mnist': FashionMNISTClassifier}[dset]
        self.classifier = Classifier(temperature=temperature)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(device)
        self.classifier.eval()

    def generate(self, z):
        print (z)
        print (type(z))
        img = self.vae.decode(z)
        return img

    def classify(self, img):
        prob = self.classifier(img)
        return prob

    def get_mask(self, img, target=None):
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()

        img.requires_grad_()

        scores = self.classifier(img)

        if target == None:
            target = scores.argmax()
        score_max = scores[0, target]

        score_max.backward()

        grad = img.grad.data[0]

        return grad


    def get_mask_2(self, img, target=None):
        img.requires_grad_()
        # self.classifier.zero_grad()

        scores = self.classifier(img)

        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]

        score_max.backward()

        saliency, _ = torch.max(img.grad.data, dim=0)

        return saliency

    def get_smoothed_mask(self, x_value, stdev_spread=0.25, nsamples=25, magnitude=False):
        """
        SOURCE: https://github.com/PAIR-code/saliency/blob/master/saliency/base.py
        Returns a mask that is smoothed with the SmoothGrad method.
        Args:
          x_value: Input value, not batched.
          feed_dict: (Optional) feed dictionary to pass to the session.run call.
          stdev_spread: Amount of noise to add to the input, as fraction of the
                        total spread (x_max - x_min). Defaults to 15%.
          nsamples: Number of samples to average across to get the smooth gradient.
          magnitude: If true, computes the sum of squares of gradients instead of
                     just the sum. Defaults to true.
        """
        stdev = stdev_spread * (torch.max(x_value) - torch.min(x_value))

        total_gradients = torch.zeros_like(x_value)
        for i in range(nsamples):
            noise = torch.empty(x_value.shape).normal_(mean=0,std=stdev).to('cuda')
            x_plus_noise = x_value + noise
            grad = self.get_mask(x_plus_noise)
            if magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad

        return total_gradients / nsamples

    # def hook_layers(self):
    #     def hook_function(module, grad_in, grad_out):
    #         self.gradients = grad_in[0]
    #
    #     # Register hook to the first layer
    #     first_layer = list(self.classifier.features._modules.items())[0][1]
    #     first_layer.register_backward_hook(hook_function)

    def saliency_map(self, img):
        #
        # scores = self.classifier(img)
        # score_max_idx = scores.argmax()
        # score_max = scores[0, score_max_idx]
        #
        # score_max.backward()
        # saliency, _ = torch.max(img.grad.data.abs(),dim=0)
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(1,4,1)

        saliency = self.get_smoothed_mask(img)
        saliency = saliency.cpu().detach().numpy()[0]
        print (saliency.shape)
        saliency = cv2.blur(saliency, (2,2))

        saliency_vanilla = self.get_mask(img)
        saliency_vanilla = saliency_vanilla.cpu().detach().numpy()
        print (saliency_vanilla.shape)

        saliency_vanilla = cv2.blur(saliency_vanilla, (2,2))

        # print (saliency.shape)

        ax1.imshow(img[0].cpu().detach().numpy(), cmap = 'gray')#, cmap=plt.cm.hot)
        ax1.axis('off')
        ax1.set_title('Input')

        ax2 = fig.add_subplot(1,4,2)
        ax2.imshow(saliency_vanilla, cmap=plt.cm.coolwarm)
        ax2.axis('off')
        ax2.set_title('Vanilla')


        ax3 = fig.add_subplot(1,4,3)
        ax3.imshow(saliency, cmap=plt.cm.coolwarm)
        ax3.axis('off')
        ax3.set_title('SmoothGrad')

        ax4 = fig.add_subplot(1,4,4)
        ax4.imshow(img[0].cpu().detach().numpy(), cmap = 'gray')
        ax4.imshow(saliency, alpha = 0.7, cmap=plt.cm.coolwarm)
        ax4.axis('off')
        ax4.set_title('SG Overlaid')
        #
        plt.show()

    def predict_from_latent(self, z, return_img=False):
        assert len(z.shape)==1, 'z must be 1 dimensional'
        img = self.generate(z)
        prob = self.classify(img).view(-1)
        if return_img:
            return prob, img.view(28, 28)
        else:
            return prob

class GANModel(NNModel):
    def __init__(self, dset, latent_dim=5, gan_path=None, classifier_path=None, temperature=1., device='cuda'):
        super(GANModel, self).__init__()
        assert dset in ['mnist', 'fashion_mnist'], 'this model only supports mnist fashion_mnist'
        self.latent_dim = latent_dim
        self.device = device

        if gan_path is None:
            print('using default GAN')
            gan_path = 'saved_models/%s_gan.pth'%dset
        GAN = {'mnist': MNISTGAN, 'fashion_mnist': FashionMNISTGAN}[dset]
        self.gan = GAN(latent_dim)
        self.gan.load_state_dict(torch.load(gan_path))
        self.gan.to(device)
        self.gan.eval()

        if classifier_path is None:
            print('using default classifier')
            classifier_path = 'saved_models/%s_classifier.pth'%dset
        Classifier = {'mnist': MNISTClassifier, 'fashion_mnist': FashionMNISTClassifier}[dset]
        self.classifier = Classifier(temperature=temperature)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(device)
        self.classifier.eval()

    def generate(self, z):
        img = self.gan(z)
        return img

    def classify(self, img):
        prob = self.classifier(img)
        return prob

    def predict_from_latent(self, z, return_img=False):
        assert len(z.shape)==1, 'z must be 1 dimensional'
        img = self.generate(z)
        prob = self.classify(img).view(-1)
        if return_img:
            return prob, img
        else:
            return prob

class ADDAModel(NNModel):
    def __init__(self, model, latent_dim=5, temperature=1., device='cuda'):
        super(ADDAModel, self).__init__()
        self.latent_dim = latent_dim
        self.device = device

        gan_path = 'saved_models/mnist_gan.pth'
        self.gan = MNISTGAN(latent_dim)
        self.gan.load_state_dict(torch.load(gan_path))
        self.gan.to(device)
        self.gan.eval()

        if model == 'baseline':
            classifier_path = 'saved_models/baseline.lenet'
        elif model == 'adda':
            classifier_path = 'saved_models/adda.lenet'
        self.classifier = LeNet(temperature=temperature)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(device)
        self.classifier.eval()

    def generate(self, z):
        img = self.gan(z)
        return img

    def classify(self, img):
        prob = self.classifier(img)
        return prob

    def predict_from_latent(self, z, return_img=False):
        assert len(z.shape)==1, 'z must be 1 dimensional'
        img = self.generate(z)
        prob = self.classify(img).view(-1)
        if return_img:
            return prob, img
        else:
            return prob
