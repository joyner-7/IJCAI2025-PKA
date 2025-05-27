from __future__ import absolute_import

from .triplet_loss_transreid import TripletLoss, SoftTripletLoss, SoftTripletLoss_weight,PlasticityLoss
from .crossentropy import CrossEntropyLabelSmooth, CrossEntropyLabelSmooth_weighted

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'CrossEntropyLabelSmooth_weighted',
    'SoftTripletLoss_weight',
    'PlasticityLoss'
]
