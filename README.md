# GPT-NeoX-Pretrain-Finetuning 

## 모델 설명

GPT-NeoX 모델은 EleutherAI의 오픈소스 거대 언어 모델(LLM)입니다. 원하는 언어로 프리트레인이 가능하며, 프리트레인된 GPT-NeoX 모델은 파인튜닝에 따라 텍스트 생성, 번역, 요약, 질의 응답 등 다양한 태스크를 수행할 수 있습니다.

## 모델 아키텍처

GPT-NeoX 모델은 GPT-3와 유사한 구조를 지닌 자동 회귀 트랜스포머 디코더 모델입니다. GPT는 Generative Pre-trained Transformer의 약자로 대규모의 데이터로 학습하여 자연어의 표현을 학습한 인공지능 모델이며, 파인튜닝을 통하여 다양한 자연어 처리 태스크를 수행할 수 있습니다. GPT 모델은 다음에 올 단어를 예측하는 비지도 학습 방식으로 학습을 하였으며, 이전에 생성된 단어의 문맥에 기반하여 새로운 단어를 생성할 수 있습니다.   

## 학습 데이터셋

● 프리트레인: Key가 "text"이고 Value는 텍스트로 이루어진 jsonl 포맷 파일
● 파인튜닝: 질문과 대답 쌍으로 이루어진 데이터셋, 혹은 Alpaca 데이터셋처럼 Instruction, Input, Output으로 이루어진 데이터셋

## 환경 설정과 의존성

프리트레인과 파인튜닝 주피터 노트북 파일은 코랩 기준으로 작성되었습니다.  
프리트레인은 파이썬은 3.8 버젼 이, PyTorch는 1.8 버젼 이상을 사용할 것을 권장합니다. 또한 윈도우에서는 DeepSpeed 라이브러리가 일부만 지원되므로, 우분투 환경을 사용할 것을 권장합니다.   

### 의존성 라이브러리 설치
```
# 프리트레인
best_download
git+https://github.com/EleutherAI/DeeperSpeed.git#egg=deepspeed
ftfy>=6.0.1
git+https://github.com/EleutherAI/lm_dataformat.git@4eec05349977071bf67fc072290b95e31c8dd836
huggingface_hub>=0.11.0
lm_eval==0.3.0
mpi4py>=3.0.3
numpy>=1.22.0
pybind11>=2.6.2
regex
sentencepiece
six
tiktoken>=0.1.2
tokenizers>=0.12.1
transformers==4.30.2

# 파인튜닝
transformers
datasets
pandas
torch 
huggingface_hub 
```

## 모델 파라미터 

configs 폴더에서 원하는 사이즈의 파라미터 파일을 선택하면 됩니다. 총 파라미터 사이즈에 영향을 주는 파라미터는 다음과 같습니다.  
● num_layers: 레이어 개수  
● hidden_size: 은닉 상태의 차원 수    
● num_attention_heads: 어텐션 헤드 개수  
● seq_length: 최대 시퀀스 길이  
● vocab_size: 토크나이저의 총 토큰 개수  

## 평가 기준

SSA(Sensibleness and Specificity Average) 스코어를 사용하였습니다. SSA는 문장이 말이 되는가를 측정하는 Sensibleness와 얼마나 대답이 구체적인지를 측정하는 Specificity 두 가지 지표를 사용하며, 0과 1로 평가합니다.    
모델의 SSA 스코어 결과는 62.5점입니다.   
