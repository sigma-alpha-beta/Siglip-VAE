## Training Scripts of SIGLIP-VAE

### Installation

1. Install the lightningdit environment first.

2. Install additional packages:
    ```
    pip install -r vavae_requirements.txt
    ```

3. [Taming-Transformers](https://github.com/CompVis/taming-transformers?tab=readme-ov-file) is also needed for training. 
    
    Get it by running:
    ```
    git clone https://github.com/CompVis/taming-transformers.git
    cd taming-transformers
    pip install -e .
    ```

    Then modify ``./taming-transformers/taming/data/utils.py`` to meet torch 2.x:
    ```
    export FILE_PATH=./taming-transformers/taming/data/utils.py
    sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' "$FILE_PATH"
    ```


### Train

1. Modify training config as you need.

2. Run training by:

    ```
    bash run_train.sh vavae/configs/f16d32_siglipv2_long.yaml
    ```
    Your training logs and checkpoints will be saved in the `logs` folder.

### Evaluate

Put your checkpoint path into ``lightningdit/tokenizer/configs/siglipvae_f16d32.yaml`` and use ``lightningdit/evaluate_tokenizer.py`` to evaluate the model.

### Acknowledgement

SIGLIP-VAE's training is mainly built upon [VA-VAE](https://github.com/hustvl/LightningDiT/tree/main/). Thanks for the great work!
