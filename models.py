import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import apex
from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP
from transformers import *

from transformers.modeling_bert import BertEncoder,BertLayer
from transformers.modeling_xlnet import XLNetLayer,XLNetRelativeAttention

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

class SplitHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.first = nn.Linear(hidden_size,30)
    def forward(self, x):
        x = torch.chunk(x,2,0)
        x = self.first(torch.cat(x,1))
        return x

def create_bert_model(model_path):
    conf = MODEL_CLASSES['bert'][0].from_pretrained(model_path)
    conf.num_labels = 30
    model = MODEL_CLASSES['bert'][1].from_pretrained(
        model_path, config=conf)
    # model.roberta.embeddings.token_type_embeddings = nn.Embedding(
    #     2, model.roberta.embeddings.token_type_embeddings.weight.shape[1])
    # model.roberta.embeddings.token_type_embeddings.weight.data.zero_()
    model.bert.encoder.forward = roberta_cls_forward.__get__(
        model.bert.encoder, BertEncoder)
    model.classifier = SplitHead(2*conf.hidden_size)
    return model

def create_roberta_model(model_path):
    conf = MODEL_CLASSES['roberta'][0].from_pretrained(model_path)
    conf.num_labels = 30
    model = MODEL_CLASSES['roberta'][1].from_pretrained(
        model_path, config=conf)
    # model.roberta.embeddings.token_type_embeddings = nn.Embedding(
    #     2, model.roberta.embeddings.token_type_embeddings.weight.shape[1])
    # model.roberta.embeddings.token_type_embeddings.weight.data.zero_()
    model.roberta.encoder.forward = roberta_cls_forward.__get__(
        model.roberta.encoder, BertEncoder)
    model.classifier.out_proj = SplitHead(2*conf.hidden_size)
    return model

def create_roberta_coatt_model(model_path,coat_start_layer=20):
    conf = MODEL_CLASSES['roberta'][0].from_pretrained(model_path)
    conf.num_labels = 30
    model = MODEL_CLASSES['roberta'][1].from_pretrained(
        model_path, config=conf)
    # model.roberta.embeddings.token_type_embeddings = nn.Embedding(
    #     2, model.roberta.embeddings.token_type_embeddings.weight.shape[1])
    # model.roberta.embeddings.token_type_embeddings.weight.data.zero_()
    for i,x in enumerate(model.roberta.encoder.layer):
        if i < coat_start_layer: continue
        x.forward = roberta_coatt_layer_forward.__get__(x, BertLayer)
    model.classifier.out_proj = SplitHead(2*conf.hidden_size)
    return model

def create_xlnet_model(model_path):
    conf = MODEL_CLASSES['xlnet'][0].from_pretrained(model_path)
    conf.num_labels = 30
    model = MODEL_CLASSES['xlnet'][1].from_pretrained(
        model_path, config=conf)
    model.transformer.forward = xlnet_cls_forward.__get__(model.transformer, XLNetLayer)
    model.logits_proj = SplitHead(2*conf.hidden_size)
    return model

def create_xlnet_coatt_model(model_path,coat_start_layer=20):
    conf = MODEL_CLASSES['xlnet'][0].from_pretrained(model_path)
    conf.num_labels = 30
    model = MODEL_CLASSES['xlnet'][1].from_pretrained(
        model_path, config=conf)
    for i,x in enumerate(model.transformer.layer):
        if i < coat_start_layer: continue
        x.rel_attn.forward = xlnet_coatt_layer_forward.__get__(x.rel_attn, XLNetRelativeAttention)
    model.logits_proj = SplitHead(2*conf.hidden_size)
    return model

def wrap_model(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6,
                            lr=1e-5)
    model = model.cuda()
    model = model.train()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0,
                                        loss_scale='dynamic')
    model = DDP(model)
    optimizer.zero_grad()
    model.zero_grad()
    return model, optimizer

def roberta_cls_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        attention_mask = torch.cat(
            [attention_mask[:, :, :, :1],
             attention_mask], -1)
        added_state = hidden_states[:, :1].flip(0)
        hidden_states = torch.cat([added_state, hidden_states], 1)
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states[:hidden_states.size(0)//2, 0] = hidden_states[hidden_states.size(0)//2:, 1]
            hidden_states[hidden_states.size(0)//2:, 0] = hidden_states[:hidden_states.size(0)//2, 1]
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        hidden_states = hidden_states[:, 1:]
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs

def xlnet_cls_forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        attention_mask = torch.cat([attention_mask[:,:1],attention_mask],-1)
        if token_type_ids is not None:
            token_type_ids = torch.cat([token_type_ids[:,:1],token_type_ids],-1)
        token_type_ids = token_type_ids.transpose(0, 1).contiguous() if token_type_ids is not None else None
        input_mask = input_mask.transpose(0, 1).contiguous() if input_mask is not None else None
        attention_mask = attention_mask.transpose(0, 1).contiguous() if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0).contiguous() if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        qlen+=1
        klen = mlen + qlen

        dtype_float = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError("Unsupported attention type: {}".format(self.attn_type))
        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatbility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz]).to(data_mask)
                data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = torch.cat([torch.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        added_state = output_h[:1].flip(1)
        output_h = torch.cat([added_state, output_h], 0)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.expand(target_mapping.shape[0], bsz, -1)
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = torch.zeros([mlen, bsz], dtype=torch.long, device=device)
                cat_ids = torch.cat([mem_pad, token_type_ids], dim=0)
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = F.one_hot(seg_mat, num_classes=2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)

        attentions = []
        hidden_states = []

        # added_state = hidden_states[:, :1].flip(0)
        # hidden_states = torch.cat([added_state, hidden_states], 1)
        for i, layer_module in enumerate(self.layer):
            if self.mem_len is not None and self.mem_len > 0 and self.output_past:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if self.output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)
            output_h[0,:output_h.size(1)//2] = output_h[1,output_h.size(1)//2:]
            output_h[0,output_h.size(1)//2:] = output_h[1,:output_h.size(1)//2]
            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
            )
            output_h, output_g = outputs[:2]
            if self.output_attentions:
                attentions.append(outputs[2])
        output_h = output_h[1:]
        # Add last hidden state
        if self.output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = (output.permute(1, 0, 2).contiguous(),)

        if self.mem_len is not None and self.mem_len > 0 and self.output_past:
            outputs = outputs + (new_mems,)

        if self.output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2).contiguous() for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2).contiguous() for hs in hidden_states)
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1).contiguous() for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs = outputs + (attentions,)

        return outputs  # outputs, (new_mems), (hidden_states), (attentions)

def roberta_coatt_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None
):
    encoder_hidden_states = torch.cat(list(reversed(torch.chunk(hidden_states, 2,0))),0)
    encoder_attention_mask = torch.cat(list(reversed(torch.chunk(attention_mask, 2,0))),0)
    self_attention_outputs = self.attention(hidden_states, attention_mask,
        head_mask, encoder_hidden_states, encoder_attention_mask)
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]
    
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + outputs
    return outputs


def xlnet_coatt_layer_forward(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None):

    cat = h
    cat = torch.cat(list(reversed(torch.chunk(cat, 2,1))),1)
    attn_mask_h = torch.cat(list(reversed(torch.chunk(attn_mask_h, 2,1))),1)
    # r = torch.cat(list(reversed(torch.chunk(r, 2,1))),1)

    # content heads
    q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)

    # positional heads
    k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)

    # core attention ops
    attn_vec = self.rel_attn_core(
        q_head_h, k_head_h, v_head_h, k_head_r, seg_mat=seg_mat, attn_mask=attn_mask_h, head_mask=head_mask
    )

    if self.output_attentions:
        attn_vec, attn_prob = attn_vec

    # post processing
    output_h = self.post_attention(h, attn_vec)
    output_g = None

    outputs = (output_h, output_g)
    if self.output_attentions:
        outputs = outputs + (attn_prob,)
    return outputs