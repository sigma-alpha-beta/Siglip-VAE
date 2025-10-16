### SIGLIP-VAE

- Modify `tokenizer/configs/siglipvae_f16d32.yaml` to use your own checkpoint path.

### Extract ImageNet Latents

- We use SIGLIP-VAE to extract latents for all ImageNet images. During extraction, we apply random horizontal flips to maintain consistency with previous works. Run:

- Modify `extract_features.py` to your own data path and {output_path}.
    
    ```
    bash run_extraction.sh tokenizer/configs/siglipvae_f16d32.yaml
    ```


### Train LightningDiT

- However, you need to modify some necessary paths as required in ``configs/lightningdit_xl_siglipvae_f16d32.yaml``.

- Run the following command to start training. It train 64 epochs with LightningDiT-XL/1.

    ```
    bash run_train.sh configs/lightningdit_xl_siglipvae_f16d32.yaml
    ```

### Inference

- Let's see some demo inference results first before we calculate FID score.

    Run the following command:
    ```
    bash run_fast_inference.sh configs/lightningdit_xl_siglipvae_f16d32.yaml
    ```
    Images will be saved into ``demo_images/demo_samples.png``

- Calculate FID score:

    ```
    bash run_fid_eval.sh configs/lightningdit_xl_siglipvae_f16d32.yaml
    ```
    It will provide a reference FID score. For the final reported FID score in the publication, you need to use ADM's evaluation code for standardized testing.
