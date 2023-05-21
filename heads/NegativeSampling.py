from OpenKE.openke.module.strategy.Strategy import Strategy
from mmdet.utils import get_root_logger, get_device
import torch
logger = get_root_logger()  

class NegativeSampling(Strategy):

	def __init__(self, model=None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	# def _get_positive_score(self, score):
	# 	score = torch.cat(score)
	# 	positive_scores = score.view(-1, score.size()[0]).permute(1, 0)
	# 	# logger.info(f'positive_score:{positive_scores}')
	# 	positive_score = torch.sum(positive_scores, dim=0)
		
	# 	return positive_score

	# def _get_negative_score(self, score):
	# 	score = torch.cat(score)
	# 	negative_scores = score.view(-1, score.size()[0]).permute(1, 0)
	# 	# logger.info(f'negative_score:{negative_scores}')
	# 	negative_score = torch.sum(negative_scores, dim=0)
	# 	return negative_score
	def _get_positive_score(self, score):
		# logger.info(f'score:{score}')
		positive_score = score[:len(score)//2]
		# logger.info(f'positive_score:{positive_score}')
		positive_score = positive_score.view(-1, len(score)//2).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[len(score)//2:]
		negative_score = negative_score.view(-1, len(score)//2).permute(1, 0)
		return negative_score

	def regularization(self,head,relation,relation_norm,tail):
		head = torch.cat(head)
		tail = torch.cat(tail)
		relation = torch.cat(relation)
		relation_norm = torch.cat(relation_norm)
		regul = (torch.mean(head ** 2) + 
				 torch.mean(tail ** 2) + 
				 torch.mean(relation ** 2) +
				 torch.mean(relation_norm ** 2)) / 4
		return regul
	def forward(self,score):
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		# logger.info(f'p_score:{p_score.size()}')
		loss_res = self.loss(p_score,n_score)
		
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.regularization(head,relation,relation_norm,tail)
		# if self.l3_regul_rate != 0:
		# 	loss_res += self.l3_regul_rate * self.model.l3_regularization(head,relation,relation_norm,tail)

		return loss_res