# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .dotav2 import DOTAv2Dataset
from .dotav1_5 import DOTAv1_5Dataset
from .dior import DIORDataset

__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset','DOTAv2Dataset','DIORDataset','DOTAv1_5Dataset']
