import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig
from helper import get_performance, get_loss_fn, GRAPH_MODEL_CLASS
from models.prompter import Prompter
from models.bert_for_layerwise import BertModelForLayerwise
from sklearn.cluster import KMeans

from models.leading_tree import LeadingTree
from models.reshape_pooling import FeatureExtractor

import torch.distributed as dist
from torch.distributed import all_gather_object


# Improved Leading Tree + Pooling
class KGCPromptTuner(pl.LightningModule):
    def __init__(self, configs, text_dict, gt):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']

        # Temperature parameter
        self.inv_t = nn.Parameter(torch.tensor(1.0 / configs.temperature).log(), requires_grad=True)

        self.ent_embed = nn.Embedding(self.configs.n_ent, self.configs.embed_dim)
        nn.init.xavier_normal_(self.ent_embed.weight.data)

        if self.configs.graph_model in ['transe', 'rotate']:
            self.rel_embed = nn.Embedding(self.configs.n_rel, self.configs.embed_dim)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            self.rel_embed = nn.Embedding(self.configs.n_rel * 2, self.configs.embed_dim)
        nn.init.xavier_normal_(self.rel_embed.weight.data)

        self.plm_configs = AutoConfig.from_pretrained(configs.pretrained_model)
        self.plm_configs.prompt_length = self.configs.prompt_length
        self.plm_configs.prompt_hidden_dim = self.configs.prompt_hidden_dim
        self.plm = BertModelForLayerwise.from_pretrained(configs.pretrained_model)

        self.prompter = Prompter(self.plm_configs, configs.embed_dim, configs.prompt_length)

        self.fc = nn.Linear(configs.prompt_length * self.plm_configs.hidden_size, configs.embed_dim)
        self.mlp_res = nn.Sequential(
            nn.Linear(configs.embed_dim, configs.embed_dim * 4),  
            nn.GELU(),  
            nn.Dropout(p=0.1),  
            nn.Linear(configs.embed_dim * 4, configs.embed_dim),  
            nn.Dropout(p=0.1)  
        )
        
        # Correct initialization: Apply Kaiming initialization only to Linear layers
        for layer in self.mlp_res:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')  
                init.zeros_(layer.bias)  

        self.feature_extractor = FeatureExtractor(self.plm_configs.num_hidden_layers, configs.prompt_length, self.plm_configs.hidden_size)
        self.gate_layer_ent = nn.Linear(self.plm_configs.hidden_size, 1)
        self.gate_layer_rel = nn.Linear(self.plm_configs.hidden_size, 1)

        if configs.prompt_length > 0:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)

        if configs.n_lar > 0:
            self.lar_loss_fn = nn.TripletMarginWithDistanceLoss(
                margin=configs.gamma,
                distance_function=lambda x, y: self.graph_model.score_fn(x, y[0], y[1]),
            )

        self.history = {'perf': ..., 'loss': []}
        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9

        # Initialize alpha and beta parameters
        self.alpha = 0.01 if self.configs.alpha_step > 0 else self.configs.alpha
        self.beta = 0.01 if self.configs.beta_step > 0 else self.configs.beta

    def select_representative_samples(self, embeddings, distance_function=None):
        """Select representative samples using leading tree clustering"""
        if distance_function is None:
            distance_function = torch.cdist

        representative_samples = []

        for batch_idx in range(embeddings.shape[0]):
            batch_samples = embeddings[batch_idx]
            D = distance_function(batch_samples, batch_samples, 2).detach().cpu().numpy()

            # Build leading tree
            lt = LeadingTree(X_train=batch_samples.detach().cpu().numpy(), dc=0.9, lt_num=2, D=D)
            lt.ComputeLocalDensity(D, dc=0.9)
            lt.ComputeParentNode(D, lt.Q)
            lt.ProCenter(lt.density, lt.delta, lt.Pa)
            lt.GetSubtreeR(lt.gamma_D, lt.lt_num, lt.Q, lt.Pa)

            # Calculate average density of each subtree and select the one with maximum density
            max_density_tree, subtree_avg_density = lt.GetSubtreeAverageDensity(lt.gamma_D, lt.lt_num, lt.Q, lt.Pa)

            # Get indices of all nodes in the maximum density subtree
            subtree_nodes = lt.AL[max_density_tree]  

            # Get embeddings of these nodes
            subtree_embeddings = batch_samples[subtree_nodes]

            # Calculate average embedding of these nodes
            representative_sample = subtree_embeddings.mean(dim=0, keepdim=True)

            # Add the calculated representative sample to the list
            representative_samples.append(representative_sample)

        # Concatenate all representative samples and return
        return torch.cat(representative_samples, dim=0).to('cuda')

    def forward(self, ent_rel, src_ids, src_mask):
        """Forward pass of the model"""
        bs = ent_rel.size(0)
        all_ent_embed = self.ent_embed.weight
        if self.configs.graph_model in ['transe', 'rotate']:
            all_rel_embed = torch.cat([self.rel_embed.weight, -self.rel_embed.weight], dim=0)
        elif self.configs.graph_model in ['null', 'conve', 'distmult']:
            all_rel_embed = self.rel_embed.weight

        ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        ent_embed = all_ent_embed[ent]
        rel_embed = all_rel_embed[rel]

        # Use Prompter to generate prompts
        prompt = self.prompter(torch.stack([ent_embed, rel_embed], dim=1))
        prompt_attention_mask = torch.ones(ent_embed.size(0), self.configs.prompt_length * 2).type_as(src_mask)
        src_mask = torch.cat((prompt_attention_mask, src_mask), dim=1)
        output = self.plm(input_ids=src_ids, attention_mask=src_mask, layerwise_prompt=prompt)

        # last_hidden_state -- .shape: (batch_size, seq_len, model_dim)
        last_hidden_state = output.last_hidden_state

        # (batch_size, prompt_length * 2, model_dim)
        ent_rel_state = last_hidden_state[:, :self.configs.prompt_length * 2]

        # (batch_size, prompt_length, hidden_size)
        plm_ent_embed, plm_rel_embed = torch.chunk(ent_rel_state, chunks=2, dim=1)

        # Get complete features from output (all layers)
        ent_out_tensor_full, rel_out_tensor_full = output.out_tensor

        # Exclude the last layer (slice the last K positions)
        ent_out_tensor = ent_out_tensor_full[:, :-self.configs.prompt_length, :]
        rel_out_tensor = rel_out_tensor_full[:, :-self.configs.prompt_length, :]

        reduced_ent_out_tensor, reduced_rel_out_tensor = self.feature_extractor(ent_out_tensor, rel_out_tensor)

        # ----------------Define gating mechanism----------------------------#
        # Calculate weights (batch_size, prompt_length, 1)
        gate_values_ent = torch.sigmoid(self.gate_layer_ent(plm_ent_embed + reduced_ent_out_tensor))
        gate_values_rel = torch.sigmoid(self.gate_layer_rel(plm_rel_embed + reduced_rel_out_tensor))

        # Feature fusion
        plm_ent_embed = gate_values_ent * plm_ent_embed + (1 - gate_values_ent) * reduced_ent_out_tensor
        plm_rel_embed = gate_values_rel * plm_rel_embed + (1 - gate_values_rel) * reduced_rel_out_tensor

        plm_ent_embed = self.fc(plm_ent_embed.reshape(ent_embed.size(0), -1))
        plm_rel_embed = self.fc(plm_rel_embed.reshape(rel_embed.size(0), -1))

        # (batch_size, embed_dim)
        plm_ent_embed = F.normalize(self.mlp_res(plm_ent_embed) + plm_ent_embed, p=2, dim=-1)
        plm_rel_embed = F.normalize(self.mlp_res(plm_rel_embed) + plm_rel_embed, p=2, dim=-1)

        # pred -- .shape: (batch_size, embed_dim)
        pred = self.graph_model(plm_ent_embed, plm_rel_embed)

        # logits -- .shape: (batch_size, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)

        return logits, pred

    def training_step(self, batched_data, batch_idx):
        """Training step with LAR loss and contrastive loss"""
        # Extract data from input batch
        if self.configs.alpha_step > 0 and self.alpha < self.configs.alpha:
            self.alpha = min(self.alpha + self.configs.alpha_step, self.configs.alpha)
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        ent_rel = batched_data['ent_rel']
        tgt_ent = batched_data['tgt_ent']
        labels = batched_data['labels']
        lars = batched_data['lars'] if self.configs.n_lar > 0 else None

        # Calculate model predictions
        logits, pred = self(ent_rel, src_ids, src_mask)
        loss = self.loss_fn(logits, labels)

        # ========================== LAR Loss Calculation ==========================
        if self.configs.n_lar > 0:
            lar_ent_embed = self.ent_embed
            pos = lar_ent_embed(labels).unsqueeze(1)
            lar_embeddings = lar_ent_embed(lars)                      # Get embeddings of negative samples
            lar = self.select_representative_samples(lar_embeddings)  # Select representative negative samples

            pos_bias = self.graph_model.bias[labels].unsqueeze(-1)
            lar_bias = torch.mean(self.graph_model.bias[lars], dim=-1, keepdim=True)

            lar_loss = self.lar_loss_fn(anchor=pred, positive=(pos, pos_bias), negative=(lar, lar_bias))
            loss += self.alpha * lar_loss

            # -----------Contrastive Loss-----------------
            if self.configs.beta_step > 0 and self.beta < self.configs.beta:
                self.beta = min(self.beta + self.configs.beta_step, self.configs.beta)
            # Calculate positive sample similarity
            pos_sim = (torch.sum(pred.unsqueeze(1) * pos, dim=-1) + pos_bias) * self.inv_t  

            # Calculate negative sample similarity
            neg_sim = (torch.matmul(pred.unsqueeze(1), lar_embeddings.permute(0, 2, 1)) + lar_bias.unsqueeze(-1)) * self.inv_t 

            # Concatenate positive and negative sample similarities
            logit = torch.cat([pos_sim, neg_sim.squeeze(1)], dim=-1)  

            # Define labels
            label = torch.zeros(logit.shape[0], dtype=torch.long, device=logit.device)  

            # Calculate InfoNCE loss
            infonce_loss = torch.nn.functional.cross_entropy(logit, label)
            loss += self.beta * infonce_loss  

        self.history['loss'].append(loss.detach().item())
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx, dataset_idx):
        """Validation step for tail/head prediction"""
        src_ids = batched_data['source_ids']
        src_mask = batched_data['source_mask']
        test_triples = batched_data['triple']
        ent_rel = batched_data['ent_rel']
        src_ent, rel = ent_rel[:, 0], ent_rel[:, 1]
        tgt_ent = batched_data['tgt_ent']
        gt = self.all_tail_gt if dataset_idx == 0 else self.all_head_gt
        logits, _ = self(ent_rel, src_ids, src_mask)
        logits = logits.detach()
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            if self.configs.is_temporal:
                tgt_filter = gt[(hi, ri, test_triples[i][3])]
            else:
                tgt_filter = gt[(hi, ri)]
            tgt_score = logits[i, ti].item()
            logits[i, tgt_filter] = self._MASKING_VALUE
            logits[i, ti] = tgt_score
        _, argsort = torch.sort(logits, dim=1, descending=True)
        argsort = argsort.cpu().numpy()

        ranks = []
        for i in range(len(src_ent)):
            hi, ti, ri = src_ent[i].item(), tgt_ent[i], rel[i].item()
            rank = np.where(argsort[i] == ti)[0][0] + 1
            ranks.append(rank)
        if self.configs.use_log_ranks:
            filename = os.path.join(self.configs.save_dir, f'Epoch-{self.current_epoch}-ranks.tmp')
            self.log_ranks(filename, test_triples, argsort, ranks, batch_idx)
        return ranks

    def validation_epoch_end(self, outs):
        """Aggregate validation results"""
        tail_ranks = np.concatenate(outs[0])
        head_ranks = np.concatenate(outs[1])

        perf = get_performance(self, tail_ranks, head_ranks)
        print('Epoch:', self.current_epoch)
        print(perf)

    def test_step(self, batched_data, batch_idx, dataset_idx):
        """Test step - same as validation step"""
        return self.validation_step(batched_data, batch_idx, dataset_idx)

    def test_epoch_end(self, outs):
        """Aggregate test results"""
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.AdamW(self.parameters(), lr=self.configs.lr)

    def log_ranks(self, filename, test_triples, argsort, ranks, batch_idx):
        """Log ranking results to file"""
        assert len(test_triples) == len(ranks), 'length mismatch: test_triple, ranks!'
        with open(filename, 'a') as file:
            for i, triple in enumerate(test_triples):
                if not self.configs.is_temporal:
                    head, tail, rel = triple
                    timestamp = ''
                else:
                    head, tail, rel, timestamp = triple
                    timestamp = ' | ' + timestamp
                rank = ranks[i].item()
                file.write(f'{head} {tail} {rel} {rank}{timestamp}\n')