# Third-party notices

This inference package instantiates a DINOv2 ViT-S/14 backbone through `timm`
and distributes a checkpoint derived from that pretrained backbone in the
separate model repository.

- DINOv2 was developed by Meta AI Research. Project:
  <https://github.com/facebookresearch/dinov2>
- The `timm` implementation was created and is maintained by Ross Wightman and
  contributors. Project: <https://github.com/huggingface/pytorch-image-models>
- The upstream `timm/vit_small_patch14_dinov2.lvd142m` model is distributed
  under Apache License 2.0. Model card:
  <https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m>

The Apache License 2.0 text shipped by `timm==1.0.20` is reproduced in
`THIRD_PARTY_LICENSES/Apache-2.0.txt`. These notices concern third-party
components and do not imply endorsement by their authors. That bundled license
governs the applicable upstream material; this notice does not declare an
additional license for repository-specific source code.
