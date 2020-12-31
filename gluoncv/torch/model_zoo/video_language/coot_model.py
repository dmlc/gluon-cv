from collections import OrderedDict
import os
import numpy as np
import torch
from torch import nn

from gluoncv.torch.utils.coot_utils import truncated_normal_fill


class MultiModalTransformer:
    def __init__(self,
                 cfg,
                 use_cuda: bool = True,
                 use_multi_gpu: bool = False):
        self.use_cuda = use_cuda
        self.use_multi_gpu = use_multi_gpu
        self.model_list = []

        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.video_pooler = Transformer(
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_POOLER,
            cfg.CONFIG.COOT_DATA.FEATURE_DIM)

        self.video_sequencer = Transformer(
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_SEQUENCER,
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_POOLER.OUTPUT_DIM)

        self.text_pooler = Transformer(
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_POOLER,
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_ENCODER.FEATURE_DIM)

        self.text_sequencer = Transformer(
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_SEQUENCER,
            cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_POOLER.OUTPUT_DIM)

        self.model_list = [
            self.video_pooler, self.video_sequencer, self.text_pooler,
            self.text_sequencer
        ]

    def encode_video(self, vid_frames, vid_frames_mask, vid_frames_len,
                     clip_num, clip_frames, clip_frames_len, clip_frames_mask):
        # compute video context
        vid_context = self.video_pooler(vid_frames, vid_frames_mask,
                                        vid_frames_len, None)
        if self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_SEQUENCER.USE_CONTEXT:
            if self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_SEQUENCER.NAME == "rnn":
                vid_context_hidden = vid_context.unsqueeze(0)
                vid_context_hidden = vid_context_hidden.repeat(
                    self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_SEQUENCER.
                    NUM_LAYERS, 1, 1)
            elif self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_SEQUENCER.NAME == "atn":
                vid_context_hidden = vid_context
            else:
                raise NotImplementedError
        else:
            vid_context_hidden = None

        # compute clip embedding
        clip_emb = self.video_pooler(clip_frames, clip_frames_mask,
                                     clip_frames_len, None)
        batch_size = len(clip_num)
        max_clip_len = torch.max(clip_num)
        clip_feat_dim = self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.VIDEO_POOLER.OUTPUT_DIM
        clip_emb_reshape = torch.zeros(
            (batch_size, max_clip_len, clip_feat_dim))
        clip_emb_mask = torch.zeros((batch_size, max_clip_len))
        clip_emb_lens = torch.zeros((batch_size, ))
        if self.use_cuda:
            clip_emb_reshape = clip_emb_reshape.cuda(non_blocking=True)
            clip_emb_mask = clip_emb_mask.cuda(non_blocking=True)
            clip_emb_lens = clip_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, clip_len in enumerate(clip_num):
            clip_emb_reshape[batch, :clip_len, :] =\
                clip_emb[pointer:pointer + clip_len, :]
            clip_emb_mask[batch, :clip_len] = 1
            clip_emb_lens[batch] = clip_len
            pointer += clip_len

        # compute video embedding
        vid_emb = self.video_sequencer(clip_emb_reshape, clip_emb_mask,
                                       clip_num, vid_context_hidden)

        #TODO: convert the return to an object class or maybe dictionary
        return (vid_emb, clip_emb, vid_context, clip_emb_reshape,
                clip_emb_mask, clip_emb_lens)

    def encode_paragraph(self, paragraph_caption_vectors,
                         paragraph_caption_mask, paragraph_caption_len,
                         sentence_num, sentence_caption_vectors,
                         sentence_caption_mask, sentence_caption_len):
        # compute paragraph context
        paragraph_context = self.text_pooler(paragraph_caption_vectors,
                                             paragraph_caption_mask,
                                             paragraph_caption_len, None)

        if self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_SEQUENCER.USE_CONTEXT:
            if self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_SEQUENCER.NAME == "rnn":
                paragraph_gru_hidden = paragraph_context.unsqueeze(0)
                paragraph_gru_hidden = paragraph_gru_hidden.repeat(
                    self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_SEQUENCER.
                    NUM_LAYERS, 1, 1)
            elif self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_SEQUENCER.NAME == "atn":
                paragraph_gru_hidden = paragraph_context
            else:
                raise NotImplementedError
        else:
            paragraph_gru_hidden = None

        # compute sentence embedding
        sentence_emb = self.text_pooler(sentence_caption_vectors,
                                        sentence_caption_mask,
                                        sentence_caption_len, None)
        batch_size = len(sentence_num)
        sentence_feat_dim = self.cfg.CONFIG.COOT_MODEL.MODEL_CONFIG.TEXT_POOLER.OUTPUT_DIM
        max_sentence_len = torch.max(sentence_num)
        sentence_emb_reshape = torch.zeros(
            (batch_size, max_sentence_len, sentence_feat_dim))
        sentence_emb_mask = torch.zeros((batch_size, max_sentence_len))
        sentence_emb_lens = torch.zeros((batch_size, ))
        if self.use_cuda:
            sentence_emb_reshape = sentence_emb_reshape.cuda(non_blocking=True)
            sentence_emb_mask = sentence_emb_mask.cuda(non_blocking=True)
            sentence_emb_lens = sentence_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, sentence_len in enumerate(sentence_num):
            sentence_emb_reshape[batch, :sentence_len, :] =\
                sentence_emb[pointer:pointer + sentence_len, :]
            sentence_emb_mask[batch, :sentence_len] = 1
            sentence_emb_lens[batch] = sentence_len
            pointer += sentence_len

        # compute paragraph embedding
        paragraph_emb = self.text_sequencer(sentence_emb_reshape,
                                            sentence_emb_mask, sentence_num,
                                            paragraph_gru_hidden)

        return (paragraph_emb, sentence_emb, paragraph_context,
                sentence_emb_reshape, sentence_emb_mask, sentence_emb_lens)

    def eval(self):
        for model in self.model_list:
            model.eval()
        torch.set_grad_enabled(False)

    def cuda(self, gpu):
        for model in self.model_list:
            self._to_device_fn(model, gpu)

    def train(self):
        for model in self.model_list:
            model.train()
        torch.set_grad_enabled(True)

    def _to_device_fn(self, model, gpu=0):
        torch.cuda.set_device(gpu)
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def get_params(self):
        params = []
        for model in self.model_list:
            params_dict = OrderedDict(model.named_parameters())
            _params = []
            for key, value in params_dict.items():
                _params += [{'params': value}]
            params.extend(_params)
        return params

    def load_checkpoint(self, ckpt: str):
        state = torch.load(str(ckpt))
        for i, model in enumerate(self.model_list):
            state_dict = state[i]
            if self.use_multi_gpu:
                newer_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert not key.startswith("module.")
                    new_key = "module." + key
                    newer_state_dict[new_key] = val
                model.load_state_dict(newer_state_dict)
            else:
                model.load_state_dict(state_dict)
            i += 1  # we do this intentionally

    def save_model(self, optimizer, epoch, cfg):
        """
        Save trained model weights.
        """
        model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                      cfg.CONFIG.LOG.EXP_NAME,
                                      cfg.CONFIG.LOG.SAVE_DIR)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        ckpt_name = "f{}_s{}_ckpt_epoch{}.pth".format(
            cfg.CONFIG.DATA.CLIP_LEN, cfg.CONFIG.DATA.FRAME_RATE, epoch)
        checkpoint = os.path.join(model_save_dir, ckpt_name)
        model_states = []
        for m in self.model_list:
            state_dict = m.state_dict()
            if self.use_multi_gpu:
                new_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert key.startswith("module.")
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                model_states.append(new_state_dict)
            else:
                model_states.append(state_dict)
        state = {
            'epoch': epoch + 1,
            'state_dict': model_states,
            'best_acc1': None,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, checkpoint)


class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features_count),
                                 requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(features_count),
                                 requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


def parse_pooler(input_dim, cfg) -> nn.Module:
    if cfg.POOLER == "atn":
        return AtnPool(input_dim, cfg.ATN_POOL_DIM, cfg.ATN_POOL_HEADS,
                       cfg.DROPOUT)
    elif cfg.POOLER == "avg":
        return AvgPool()
    elif cfg.POOLER == "max":
        return MaxPool()
    else:
        raise ValueError(f"unknown pooler {cfg.POOLER}")


class Transformer(nn.Module):
    def __init__(self, cfg, feature_dim: int):
        super(Transformer, self).__init__()

        self.input_norm = LayerNormalization(feature_dim)
        self.input_fc = None
        input_dim = feature_dim

        if cfg.INPUT_FC:
            self.input_fc = nn.Sequential(
                nn.Linear(feature_dim, cfg.INPUT_FC_OUTPUT_DIM), nn.GELU())
            input_dim = cfg.INPUT_FC_OUTPUT_DIM
        self.embedding = PositionalEncoding(input_dim,
                                            cfg.DROPOUT,
                                            max_len=1000)

        self.tf = TransformerEncoder(cfg.NUM_LAYERS, input_dim, cfg.NUM_HEADS,
                                     input_dim, cfg.DROPOUT)

        self.use_context = cfg.USE_CONTEXT
        if self.use_context:
            self.tf_context = TransformerEncoder(cfg.ATN_CTX_NUM_LAYERS,
                                                 input_dim,
                                                 cfg.ATN_CTX_NUM_HEADS,
                                                 input_dim, cfg.DROPOUT)

        self.pooler = parse_pooler(input_dim, cfg)

        init_network(self, init_std=0.01)

    def forward(self, features, mask, lengths, hidden_state):
        features = self.input_norm(features)
        if self.input_fc is not None:
            features = self.input_fc(features)
        features = self.embedding(features)
        features = self.tf(features, features, features, mask)
        add_after_pool = None
        if self.use_context:
            hidden_state = hidden_state.unsqueeze(1)
            ctx = self.tf_context(hidden_state, features, features, mask)
            add_after_pool = ctx.squeeze(1)
        pooled = self.pooler(features, mask, lengths)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        return pooled


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000**(2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, model_dim, heads_count, fc_dim,
                 dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, heads_count, fc_dim,
                                    dropout_prob) for _ in range(layers_count)
        ])

    def forward(self, query, key, value, mask):
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        mask = mask == 1
        sources = None

        sources = self.encoder_layers[0](query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, heads_count, fc_dim, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, model_dim, dropout_prob),
            model_dim)
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(fc_dim, model_dim, dropout_prob),
            model_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, model_dim):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(model_dim)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, model_dim, dropout_prob):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % heads_count == 0,\
            f"model dim {model_dim} not divisible by {heads_count} heads"
        self.d_head = model_dim // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.key_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.value_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.final_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, model_dim = query.size()
        d_head = model_dim // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, model_dim = key_projected.size()
        batch_size, value_len, model_dim = value_projected.size()
        query_heads = query_projected.view(batch_size, query_len,
                                           self.heads_count,
                                           d_head).transpose(1, 2)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count,
                                       d_head).transpose(1, 2)
        value_heads = value_projected.view(batch_size, value_len,
                                           self.heads_count,
                                           d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(
                mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(batch_size, query_len, model_dim)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, fc_dim, model_dim, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(model_dim, fc_dim),
                                          nn.Dropout(dropout_prob), nn.GELU(),
                                          nn.Linear(fc_dim, model_dim),
                                          nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)


class AvgPool(nn.Module):
    def forward(self, features, mask, lengths):
        _ = mask
        len_div = lengths.unsqueeze(-1).float()
        result_sum = torch.sum(features, dim=1)
        result = result_sum / len_div
        return result


class MaxPool(nn.Module):
    def forward(self, features, mask, lengths):
        result_max, _ = torch.max(features, dim=1)
        return result_max


class AtnPool(nn.Module):
    def __init__(self, d_input, d_attn, n_heads, dropout_prob):
        super(AtnPool, self).__init__()
        self.d_head = d_attn // n_heads
        self.d_head_output = d_input // n_heads
        self.num_heads = n_heads

        def _init(tensor_):
            tensor_.data = (truncated_normal_fill(tensor_.data.shape,
                                                  std=0.01))

        w1_head = torch.zeros(n_heads, d_input, self.d_head)
        b1_head = torch.zeros(n_heads, self.d_head)
        w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
        b2_head = torch.zeros(n_heads, self.d_head_output)
        _init(w1_head)
        _init(b1_head)
        _init(w2_head)
        _init(b2_head)
        self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
        self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
        self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
        self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_temp = 1
        self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=False)

    def extra_repr(self) -> str:
        strs = []
        for p in [
                self.genpool_w1_head, self.genpool_b1_head,
                self.genpool_w2_head, self.genpool_b2_head
        ]:
            strs.append(f"pool linear {p.shape}")
        return "\n".join(strs)

    def forward(self, features, mask, lengths):

        batch_size, seq_len, input_dim = features.shape
        b1 = torch.matmul(features.unsqueeze(1),
                          self.genpool_w1_head.unsqueeze(0))
        b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
        b1 = self.activation(self.dropout1(b1))
        b1 = torch.matmul(b1, self.genpool_w2_head.unsqueeze(0))
        b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
        b1 = self.dropout2(b1)
        b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)

        smweights = self.softmax(b1 / self.softmax_temp)
        smweights = self.dropout3(smweights)
        smweights = smweights.transpose(1, 2).reshape(-1, seq_len, input_dim)
        return (features * smweights).sum(
            dim=1)  # pool features with attention weights


def _init_weight(w, init_gain=1):
    w.copy_(truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):
    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            _init_weight(val.data, init_std)
