import torch
import torch.nn as nn
import torch.nn.functional as F
from OpenKE.openke.module.model.Model import Model
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger() 

class TransH(Model):

	def __init__(self, ent_tot, rel_tot, dim = 50, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransH, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.device = 'cuda'

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.norm_vector = nn.Embedding(self.rel_tot, self.dim)


		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.norm_vector.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.norm_vector.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
			
		score = torch.norm(score, self.p_norm, -1).flatten()
		# score = torch.norm(score, self.p_norm, -1)
		# logger.info(f'score in transH:{score}')
		return score

	def _transfer(self, e, norm):
		norm = F.normalize(norm, p = 2, dim = -1)
		if e.shape[1] != norm.shape[1]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			# logger.info(f'norm:{norm.size()}')
			e = e - torch.sum(e * norm, -1, True) * norm
			# logger.info(f'e:{e.size()}')
			return e.view(-1, e.shape[-1])
		else:
			return e - torch.sum(e * norm, -1, True) * norm

	def forward(self, head, relation, tail):
		mode = 'normal'
		h_ = self.ent_embeddings(head)
		# logger.info(f'h_:{h_.size()}')
		t_ = self.ent_embeddings(tail)
		# logger.info(f't_:{t_.size()}')
		r = self.rel_embeddings(relation)
		# logger.info(f'r:{r[0]},{r[1]}')
		r_norm = self.norm_vector(relation)
		h = self._transfer(h_, r_norm)
		# logger.info(f'h:{h.size()}')
		t = self._transfer(t_, r_norm)
		# logger.info(f't:{t.size()}')
		score = self._calc(h ,t, r, mode)
		# logger.info(f'embedding size of h,t,r:{h.size()},{r.size()},{t.size()}')

		if self.margin_flag:
			return self.margin - score
		else:
			return h_,r,r_norm,t_,score

	def regularization(self,head,relation,tail):
		batch_h = head
		batch_t = tail
		batch_r = relation
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_norm ** 2)) / 4
		return regul
		
	def l3_regularization(self,head,relation,tail):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)
	
	def predict(self,head,relatin,tail):
		h,t,score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return h,t,score
		else:
			return h,t,score

