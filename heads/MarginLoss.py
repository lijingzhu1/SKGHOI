import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/OpenKE/openke/module/loss')
from Loss import Loss
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger() 

class MarginLoss(Loss):

	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			# distance = torch.cdist(p_score,n_score)
			loss = ((torch.max(p_score-n_score, -self.margin)).mean() + self.margin).to('cuda')
			# loss = ((torch.max(p_score)).mean()).to('cuda')
			# logger.info(f'loss:{loss}')
			return loss
			
	
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()