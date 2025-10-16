import torch
import torch.nn as nn
from einops import rearrange

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
import torch.nn.functional as F

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", pp_style=False, vf_weight=1e2, adaptive_vf=False,
                 cos_margin=0, distmat_margin=0, distmat_weight=1.0, cos_weight=1.0) :

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.pp_style = pp_style
        if pp_style:
            print("Using pp_style for nll loss")
        self.vf_weight = vf_weight
        self.ct_weight = vf_weight # 0.5->1
        self.adaptive_vf = adaptive_vf
        self.cos_margin = cos_margin
        self.distmat_margin = distmat_margin


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def calculate_adaptive_weight_vf(self, nll_loss, vf_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            vf_grads = torch.autograd.grad(vf_loss, self.last_layer[0], retain_graph=True)[0]

        vf_weight = torch.norm(nll_grads) / (torch.norm(vf_grads) + 1e-4)
        vf_weight = torch.clamp(vf_weight, 0.0, 1e8).detach()
        vf_weight = vf_weight * self.vf_weight
        return vf_weight
    
    def calculate_adaptive_weight_ct(self, nll_loss, ct_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            ct_grads = torch.autograd.grad(ct_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            ct_grads = torch.autograd.grad(ct_loss, self.last_layer[0], retain_graph=True)[0]

        ct_weight = torch.norm(nll_grads) / (torch.norm(ct_grads) + 1e-4)
        ct_weight = torch.clamp(ct_weight, 0.0, 1e8).detach()
        ct_weight = ct_weight * self.ct_weight
        return ct_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None, z=None, aux_feature=None, z_orig=None, text_features_proj=None, logit_scale=None, enc_last_layer=None):
        if not self.pp_style:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            # kl_loss = posteriors.kl()
            # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            weighted_nll_loss = torch.mean(weighted_nll_loss)
            nll_loss = torch.mean(nll_loss)
            kl_loss = posteriors.kl(no_sum=True)
            kl_loss = torch.mean(kl_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            # vf loss
            if z is not None and aux_feature is not None:
                z_flat = rearrange(z, 'b c h w -> b c (h w)')
                aux_feature_flat = rearrange(aux_feature, 'b c h w -> b c (h w)')
                z_norm = torch.nn.functional.normalize(z_flat, dim=1)
                aux_feature_norm = torch.nn.functional.normalize(aux_feature_flat, dim=1)
                z_cos_sim = torch.einsum('bci,bcj->bij', z_norm, z_norm)
                aux_feature_cos_sim = torch.einsum('bci,bcj->bij', aux_feature_norm, aux_feature_norm)
                diff = torch.abs(z_cos_sim - aux_feature_cos_sim)
                vf_loss_1 = torch.nn.functional.relu(diff-self.distmat_margin).mean()
                vf_loss_2 = torch.nn.functional.relu(1 - self.cos_margin - torch.nn.functional.cosine_similarity(aux_feature, z)).mean()
                vf_loss = vf_loss_1*self.distmat_weight + vf_loss_2*self.cos_weight
            else:
                vf_loss = None
            
            # contrastive loss
            if z_orig is not None and text_features_proj is not None:
                image_features = z_orig.mean(dim=[2, 3]) # global average pooling -> [b, C]
                text_features = text_features_proj # [b, C]
                
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                # clip loss
                logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
                logits_per_image = logits_per_text.t()

                ct_loss = clip_loss(logits_per_text)
                try:
                    ct_weight = self.calculate_adaptive_weight_ct(nll_loss, ct_loss, last_layer=enc_last_layer)
                except RuntimeError:
                    assert not self.training
                    ct_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if vf_loss is not None:
                if self.adaptive_vf:
                    try:
                        vf_weight = self.calculate_adaptive_weight_vf(nll_loss, vf_loss, last_layer=enc_last_layer)
                    except RuntimeError:
                        assert not self.training
                        vf_weight = torch.tensor(0.0)
                else:
                    vf_weight = self.vf_weight
                
                if ct_loss is not None:
                    loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + vf_weight * vf_loss + ct_weight*ct_loss
                else:
                    loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + vf_weight * vf_loss
            else:
                if ct_loss is not None:
                    loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + ct_weight*ct_loss
                else:
                    loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            if vf_loss is not None:
                log["{}/vf_loss".format(split)] = vf_loss.detach().mean()
                if not isinstance(vf_weight, float):
                    log["{}/vf_weight".format(split)] = vf_weight.detach()
                else:
                    log["{}/vf_weight".format(split)] = torch.tensor(vf_weight)
            
            if ct_loss is not None:
                log["{}/ct_loss".format(split)] = ct_loss.detach().mean()
                if not isinstance(ct_weight, float):
                    log["{}/ct_weight".format(split)] = ct_weight.detach()
                else:
                    log["{}/ct_weight".format(split)] = torch.tensor(ct_weight)
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

