import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings : 
        Token Embeddings : Token 정보
        Position Embeddings : 위치 정보
        Segment Ebeddings : 여러 문장 입력 시 문장 구분에 사용
    """
    def __init__(self, vocab_size, embed_size, max_len, dropout_rate = 0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param max_len : max_len of sequence
        :param dropout: dropout rate
        """
        super(BERTEmbeddings, self).__init__()
        # [0] Token은 padding에 사용되기 때문에 Embedding 계산하지 않음.
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.position_embeddings = nn.Embedding(max_len, embed_size)
        # 어느 문장에 속하는 지 Token 정보 저장. 이때, [0] 토큰은 무시 됨.
        self.segment_embeddings = nn.Embedding(3, embed_size, padding_idx = 0)
        # layer_norm + dropout
        self.layer_norm = nn.LayerNorm(embed_size, eps = 1e-12)
        self.dropout = nn.Dropout(p = dropout_rate)
        #self.embed_size = embed_size -> 안 쓰일듯
    
    def forward(self, seq, segment_label = None):
        # seq : (batch, seq_len)
        batch_size ,seq_length = seq.size(0) ,seq.size(1)
        # position-ids : idx로 활용되기 때문에 torch.long으로 생성
        position_ids = torch.arange(seq.size(1), dtype=torch.long, device = seq.device)
        # position_ids를 (batch_size, seq_length) 형태로 변경.
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)            
        # token , position embeddings, segment embeddings
        token_embeddings = self.token_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        # bert_embedddings
        embeddings = token_embeddings + position_embeddings
        # segment_label이 있는 경우 segment embeddings까지 포함
        if segment_label is not None:
            segment_embeddings = self.segment_embeddings(segment_label)
            embeddings += segment_embeddings
        # layer-norm + drop out
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

    
class MultiHeadedAttention(nn.Module):
    """
    head의 수와 hidden_dim을 입력으로 받아 Multi-Head Attention 수행
    """
    def __init__(self, head_num, hidden_dim, dropout_rate_attn = 0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert hidden_dim % head_num == 0
        
        self.hidden_dim = hidden_dim
        # V와 K가 같은 경우로 진행.
        # 각 head의 dim 차원
        self.head_dim = hidden_dim / head_num
        self.head_num = head_num
        # Q,K,V에 적용될 Linear Layer
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        # 스코어 scale 변환용
        self.scale = torch.sqrt(self.head_dim)
        # dropout
        self.dropout = nn.Dropout(p = dropout_rate_attn)
        # 출력 레이어
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, q, k, v, mask = None):
        # query, key, value 모두 동일.
        # hidden_states :[batch_size, seq_length, hidden_dim(embedding)] -> embedding 레이어를 거쳐서 hidden dim이 추가 됨.
        batch_size = q.size(0)
        # Q, K, V 
        # query(key, value) : [batch_size, seq_length, hidden_dim]
        query = self.query_linear(q)
        key = self.key_linear(k)
        value = self.value_linear(v)
        # head로 분리
        # [batch, len, head 수, head 차원] -> [batch, head_num, len, head_dim]
        query = query.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # attention 스코어 계산
        # scores : [batch, head_num, query_len, key_len]
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        # mask 진행
        # mask가 False인 경우 0과 같아지므로 False인 부분에만 Masking이 된다.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        # attention 계산(각 단어에 대한 확률 계산)
        # attention : [batch, head_num, query_len, key_len]
        attention = F.softmax(scores, dim = -1)
        attention = self.dropout(attention)
        # attention과 value로 output 계산
        # output : [batch, head_num, query_len, head_dim]
        # view를 실행하기 위해서 contiguous()
        attention_seq = torch.matmul(attention, value).contiguous()
        # output : [batch, query_len, hidden_dim]
        attention_seq = attention_seq.view(batch_size, -1, self.hidden_dim)
        attention_seq = self.output_layer(attention_seq)
        
        # attention은 시각화 및 분석에 활용 가능
        return attention_seq, attention
    

class SublayerConnection(nn.Module):
    """
    현재 layer와 sublayer를 연결
    """
    def __init__(self, hidden_dim, dropout_rate = 0.1):
        super(SublayerConnection, self).__init__() 
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, layer, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.layer_norm(layer + self.dropout(sublayer))

    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_dim, ff_dim, dropout_rate = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.feed_forward_1 = nn.Linear(hidden_dim, ff_dim) # Feed-Forward 첫번째
        self.feed_forward_2 = nn.Linear(ff_dim, hidden_dim) # Feed Forward 두번째
        self.dropout = nn.Dropout(p = dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.feed_forward_2(self.dropout(self.activation(self.feed_forward_1(x))))
    

class TransformerEncoder(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden_dim, head_num, ff_dim, dropout_rate = 0.1, dropout_rate_attn = 0.1):
        """
        :param hidden_dim: hidden dim of transformer
        :param head_num: head sizes of multi-head attention
        :param feed_forward_dim: feed_forward_hidden, usually 4*hidden_dim
        :param dropout_rate: dropout rate
        :param dropout_rate_attn : attention layer의 dropout rate
        """        
        super(TransformerEncoder, self).__init__()
        # multi-head attn
        self.attention = MultiHeadedAttention(head_num = head_num, hidden_dim = hidden_dim, dropout_rate_attn = dropout_rate_attn)
        # sublayer connection - 1 (input embeddings + input embeddings attn)
        self.input_sublayer = SublayerConnection(size=hidden_dim, dropout_rate = dropout_rate)
        # FFN
        self.feed_forward = PositionwiseFeedForward(hidden_dim = hidden_dim, feed_forward_dim = ff_dim, dropout_rate = dropout_rate)
        # sublayer connection - 2 (sublayer connection 1의 결과 + Feed Forward)
        self.output_sublayer = SublayerConnection(size=hidden_dim, dropout_rate = dropout_rate)
        # self.dropout = nn.Dropout(p = dropout_rate) -> 안 쓰이는 듯
    
    def forward(self, seq, mask):
        # attention 결과
        # Q, K, V는 전부 seq 입력. attention 확률은 따로 안 쓰므로 _ 로 받아옴.
        attention_seq, _ = self.attention(q = seq, k = seq, v = seq, mask = mask)
        # sublayer connection 1 진행 : seq embeddings과 attention 결합
        connected_layer = self.input_sublayer(seq, attention_seq)
        # sublayer connection 2 진행 : conncted_layer와 FFN을 통과한 connected_layer 결합
        return self.output_sublayer(connected_layer, self.feed_forward(connected_layer))

    
class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size = 30522, max_len = 512, hidden_dim = 768, layer_num = 12, head_num = 12, dropout_ratio = 0.1, dropout_ratio_attn = 0.1):
        """
        :param vocab_size: vocab_size of total words
        :max_len : max len of seq
        :param hidden_dim: BERT model hidden size
        :param layer_num: numbers of Transformer blocks(layers)
        :param head_num : number of attention heads
        :param dropout_ratio : dropout rate
        :param dropout_ratio_attn : attention layer의 dropout ratio
        """
        super(BERT, self).__init__()
        # 기존 item에 [PAD], [MASK] 토큰이 추가 돼 item 개수 + 2
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        # encoder layer의 수
        self.layer_num = layer_num
        # head_num
        self.head_num = head_num
        self.head_dim = hidden_dim / head_num
        # dropout 비율
        self.dropout_ratio = dropout_ratio
        self.dropout_ratio_attn = dropout_ratio_attn
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.ff_dim = hidden_dim * 4
        # embedding
        self.embedding = BERTEmbeddings(vocab_size = self.vocab_size, embed_size = self.hidden_dim, max_len = self.max_len)
        # Transformer Encoder 
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(hidden_dim = self.hidden_dim, head_num = self.head_num, ff_dim = self.ff_dim, \
                                dropout_ratio = self.dropout_ratio, dropout_ratio_attn = self.dropout_ratio_attn) for _ in range(self.layer_num)])

    def forward(self, seq, segment_info = None):
        # attention masking for padded token
        # seq엔 이미 zero padding이 된 상태
        # mask : [batch_size, seq_len] -> [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
        # 브로드 캐스팅 이용
        mask = (seq > 0).squeeze(1).squeeze(1)
        # embedding the indexed sequence to sequence of vectors
        seq = self.embedding(seq, segment_info)
        # running over multiple transformer blocks
        for transformer in self.transformer_encoders:
            seq = transformer(seq, mask)

        return seq