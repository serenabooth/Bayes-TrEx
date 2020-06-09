# Bayes-Probe Experiments with CLEVR

For our experiments with CLEVR, we use and extend two repos distributed by the original CLEVR authors (Johnson et al.):

1. For Inferring and Executing Programs for Visual Reasoning:  https://github.com/facebookresearch/clevr-iep

2. For rendering new CLEVR scenes:  https://github.com/facebookresearch/clevr-dataset-gen

Our primary modifications are added to:
1. `clevr-dataset-get/image_generation/render_images.py`

    We modify this rendering code to render a scene specified by a latent

2. `clevr-iep/scripts/run_model.py`

    We modify the inference code to additionally report a prediction confidence

## Running this Code

1. You will need to download and install Blender. Follow the instructions here:
   https://github.com/serenabooth/Bayes-TrEx/tree/master/CLEVR/clevr-dataset-gen
   You may need to add a .pth file to the site-packages of Blender's
   bundled python with a command like this:

   `echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth
   `

2. You will need to download the pretrained CLEVR models. To do so:
   `cd clevr-iep;
    bash scripts/download_pretrained_models.sh
    `

3. You may need to install some dependencies:
   `pip3 install requirements.txt
   `

4. To run this code, you can start with the demo script.
   The demo script has examples for finding high-confidence examples;
   in-distribution adversarial examples; and novel-class examples.
   In addition, the demo script has code for creating saliency maps using the specified inputs.
   `cd clevr-iep; python3 scripts/demo.sh
   `


## Breakdown of scripts/demo.sh

In `clevr-iep/scripts/demo.sh`, you will find queries of the form:

```request_template = 'python3 scripts/analyze_level_sets.py \
                    --program_generator models/CLEVR/program_generator_18k.pt \
                    --execution_engine models/CLEVR/execution_engine_18k.pt \
                     --question "%s" \
                     --prob_test %s \
                     --out_of_distribution %s \
                     --class_a %s \
                     --class_b %s \
                     --save_dir %s \
                     --remove_object_type %s \
                     --num_iters %s'
```

These queries can be modified, as shown in the script, to meet the needs of each
use case (high-confidence examples, in-distribution adversarial examples, and novel
class examples).

In addition, `clevr-iep/scripts/demo.sh` has a saliency map demo. Saliency maps can be
generated with queries of the form:
```
python3 scripts/run_model.py   --program_generator models/CLEVR/program_generator_18k.pt   --execution_engine models/CLEVR/execution_engine_18k.pt --saliency_map 1  --image img/5cones.png   --question "How many cubes?"
```

The inference code is contained in `clevr-iep/scripts/analyze_level_sets.py`.

Output is saved to `output/`. In each results folder, we create an `accepted_latents.txt` file, which consists of the respective lists of accepted proposals and their corresponding probabilities.
