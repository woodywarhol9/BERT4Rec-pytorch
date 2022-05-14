# BERT4Rec_pytorch

BERT4Rec 구현하기   
https://arxiv.org/abs/1904.06690

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
├── model   
│   ├── bert.py   
│   └── __init__.py   
├── README.md  
└── trainer.py
main.py              ->구성중
requirements.txt     ->구성중    
```
- Pytorch lightning을 사용하기 위해서 module화 진행
