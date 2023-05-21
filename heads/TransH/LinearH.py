import torch
import torch.nn as nn
import torch.nn.functional as F
from pocket.ops import Flatten
from OpenKE.openke.module.model.Model import Model
from mmdet.utils import get_root_logger, get_device
logger = get_root_logger() 

class Entity_Encoder(nn.Module):
    
    def __init__(self, entity_size = 1024,entity_encoding_size = 512, head_out_size = 200):
        super().__init__()
        self.box_head = nn.Sequential(
            nn.Linear(entity_size, entity_encoding_size),
            nn.ReLU(),
            nn.Linear(entity_encoding_size, head_out_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.box_head(x)
        return x

class Relation_Encoder(nn.Module):
    
    def __init__(self, relation_size = 1024, relation_encoding_size_1 = 512,relation_encoding_size_2 = 256,relation_out_size = 200):
    	super().__init__()
    	self.spatial_head = nn.Sequential(
            nn.Linear(relation_size, relation_encoding_size_1),
            nn.ReLU(),
            nn.Linear(relation_encoding_size_1, relation_encoding_size_2),
            nn.ReLU(),
            nn.Linear(relation_encoding_size_2, relation_out_size),
            nn.ReLU()
			)  
    def forward(self, x):
    	x = self.spatial_head(x)
    	return x

class Normal_Encoder(nn.Module):
    
    def __init__(self, relation_size = 1024, relation_encoding_size_1 = 512,relation_encoding_size_2 = 256,relation_out_size = 200):
    	super().__init__()
    	self.spatial_head = nn.Sequential(
            nn.Linear(relation_size, relation_encoding_size_1),
            nn.ReLU(),
            nn.Linear(relation_encoding_size_1, relation_encoding_size_2),
            nn.ReLU(),
            nn.Linear(relation_encoding_size_2, relation_out_size),
            nn.ReLU()
			)      
    def forward(self, x):
    	x = self.spatial_head(x)
    	return x


class LinearH(Model):

	def __init__(self, ent_tot, rel_tot, entity_size = 1024, relation_size = 1024, entity_encoding_size = 512,relation_encoding_size_1 = 512,
					relation_encoding_size_2 = 256 ,trans_dim = 200,p_norm = 1, 
					norm_flag = True, margin = None, epsilon = None):
		super(LinearH, self).__init__(ent_tot, rel_tot)
		self.trans_dim = trans_dim
		# self.out_channels = out_channels
		self.entity_size = entity_size
		self.relation_size = relation_size
		self.entity_encoding_size = entity_encoding_size
		self.relation_encoding_size_1 = relation_encoding_size_1
		self.relation_encoding_size_2 = relation_encoding_size_2

		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.device = 'cuda'
		self.ent_embeddings = Entity_Encoder(self.entity_size,self.entity_encoding_size,self.trans_dim)
		self.rel_embeddings = Relation_Encoder(self.relation_size,self.relation_encoding_size_1,self.relation_encoding_size_2,self.trans_dim)
		self.norm_vector = Normal_Encoder(self.relation_size,self.relation_encoding_size_1,self.relation_encoding_size_2,self.trans_dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform(self.ent_embeddings.box_head[0].weight)
			nn.init.xavier_uniform(self.ent_embeddings.box_head[2].weight)
			nn.init.xavier_uniform_(self.rel_embeddings.spatial_head[0].weight)
			nn.init.xavier_uniform_(self.rel_embeddings.spatial_head[2].weight)
			nn.init.xavier_uniform_(self.rel_embeddings.spatial_head[4].weight)
			nn.init.xavier_uniform_(self.norm_vector.spatial_head[0].weight)
			nn.init.xavier_uniform_(self.norm_vector.spatial_head[2].weight)
			nn.init.xavier_uniform_(self.norm_vector.spatial_head[4].weight)		
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
	        # m.bias.data.fill_(0.01)

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
		# logger.info(f't_:{t_[0]},{t_[-1]}')
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
			return score

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

