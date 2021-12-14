"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

from .gan_discriminator_loss import GANDiscriminatorLoss
from .gan_generator_loss import GANGeneratorLoss
from .gan_modified_generator_loss import GANModifiedGeneratorLoss

from .wgan_discriminator_loss import WGANDiscriminatorLoss
from .wgan_generator_loss import WGANGeneratorLoss
from .wgan_grad_penalty_loss import WGANGradPenaltyLoss