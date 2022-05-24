# BERT4Rec_pytorch

BERT4Rec 구현하기   
https://arxiv.org/abs/1904.06690

모델 구조
```sh
BERT4REC(
  (model): BERT(
    (embedding): BERTEmbeddings(
      (token_embeddings): Embedding(3708, 256, padding_idx=0)
      (position_embeddings): Embedding(100, 256)
      (segment_embeddings): Embedding(3, 256, padding_idx=0)
      (layer_norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer_encoders): ModuleList(
      (0): TransformerEncoder(
        (attention): MultiHeadedAttention(
          (query_linear): Linear(in_features=256, out_features=256, bias=True)
          (key_linear): Linear(in_features=256, out_features=256, bias=True)
          (value_linear): Linear(in_features=256, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (output_linear): Linear(in_features=256, out_features=256, bias=True)
        )
        (input_sublayer): SublayerConnection(
          (layer_norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (feed_forward_1): Linear(in_features=256, out_features=1024, bias=True)
          (feed_forward_2): Linear(in_features=1024, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): GELU()
        )
        (output_sublayer): SublayerConnection(
          (layer_norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): TransformerEncoder(
        (attention): MultiHeadedAttention(
          (query_linear): Linear(in_features=256, out_features=256, bias=True)
          (key_linear): Linear(in_features=256, out_features=256, bias=True)
          (value_linear): Linear(in_features=256, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (output_linear): Linear(in_features=256, out_features=256, bias=True)
        )
        (input_sublayer): SublayerConnection(
          (layer_norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (feed_forward): PositionwiseFeedForward(
          (feed_forward_1): Linear(in_features=256, out_features=1024, bias=True)
          (feed_forward_2): Linear(in_features=1024, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (activation): GELU()
        )
        (output_sublayer): SublayerConnection(
          (layer_norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (out): Linear(in_features=256, out_features=3708, bias=True)
```

프로젝트 구조
```sh
├── data
│   ├── dataset.py
│   ├── __init__.py
│   └── ml-1m
│       ├── movies.dat
│       ├── ratings.dat
│       ├── README
│       └── users.dat
├── datamodule.py
├── lit_model.py
├── model
│   ├── bert.py
│   └── __init__.py
├── README.md
├── requirements.txt
└── trainer.py
```

- dataset.py : 전처리 및 Train/Valid/Test 데이터셋 생성
- datamodule.py : Pytorch lightning의 DataModule 역할. DataLoader 생성
- bert.py : BERT 모델 구성
- lit_model.py : Pytorch lightning의 ModelModule 역할. MLM 수행
- trainer.py - args 입력 받아 훈련 시작

수정 중
