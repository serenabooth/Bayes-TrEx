import os


for i in range(0, 10):
    model_path = "saved_models/all_but_one/fashion_mnist_vae_wo_" + str(i) + ".pth"
    save_dir = "files/fashion_mnist_vae_wo" + str(i) +"/"

    request = "python3 write_subset.py \
        --nn-model vae \
        --nn-model-path %s \
        --dataset fashion_mnist \
        --save-dir %s " % (model_path, save_dir)
    os.system(request)

    model_path = "saved_models/all_but_one/fashion_mnist_gan_wo_" + str(i) + ".pth"
    save_dir = "files/fashion_mnist_gan_wo" + str(i) + "/"

    request = "python3 write_subset.py \
        --nn-model gan \
        --nn-model-path %s \
        --dataset fashion_mnist \
        --save-dir %s " % (model_path, save_dir)
    os.system(request)
