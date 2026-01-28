# 스크립트가 위치한 디렉토리 경로를 가져오기 위한 모듈
import os

# PyTorch 딥러닝 프레임워크 - 텐서 연산 및 GPU 가속 지원
import torch

# 오디오 파일 로딩 및 처리를 위한 라이브러리 (FFmpeg 의존성 없이 mp3 로드 가능)
import librosa

# Hugging Face Transformers에서 Whisper 모델과 프로세서 클래스 임포트
# AutoModelForSpeechSeq2Seq: 음성-텍스트 변환을 위한 Seq2Seq 모델
# AutoProcessor: 오디오 전처리(특성 추출) 및 텍스트 후처리(디코딩) 담당
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# GPU(CUDA)가 사용 가능하면 "cuda:0" (첫 번째 GPU), 아니면 CPU 사용
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# GPU 사용 시 float16(반정밀도)로 메모리 절약 및 속도 향상, CPU는 float32(단정밀도) 사용
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Hugging Face Hub에서 다운로드할 모델 ID
# whisper-large-v3-turbo: OpenAI Whisper 대규모 모델의 최적화 버전
model_id = "openai/whisper-large-v3-turbo"

# 사전 학습된 Whisper 모델 로드
# dtype: 모델 가중치의 데이터 타입 지정 (torch_dtype은 deprecated)
# low_cpu_mem_usage: 모델 로딩 시 CPU 메모리 사용량 최소화
# use_safetensors: 안전한 텐서 형식 사용 (보안 및 로딩 속도 향상)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

# 모델을 지정된 디바이스(GPU 또는 CPU)로 이동
model.to(device)

# 오디오 전처리를 위한 프로세서 로드
# feature_extractor: 오디오를 멜 스펙트로그램으로 변환
# tokenizer: 생성된 토큰 ID를 텍스트로 디코딩
processor = AutoProcessor.from_pretrained(model_id)

# 이 스크립트 파일이 위치한 디렉토리에서 sample.mp3 파일의 전체 경로 생성
# __file__: 현재 스크립트의 경로
# os.path.dirname: 디렉토리 경로 추출
# os.path.join: 경로와 파일명을 OS에 맞게 결합
sample = os.path.join(os.path.dirname(__file__), "sample.mp3")

# librosa로 오디오 파일 로드
# sr=16000: Whisper 모델이 요구하는 16kHz 샘플링 레이트로 리샘플링
# audio: 오디오 데이터 (numpy 배열, 모노 채널)
# sr: 실제 샘플링 레이트 (여기서는 16000)
audio, sr = librosa.load(sample, sr=16000)

# 오디오 데이터를 모델 입력 형식으로 변환
# processor(): 오디오를 80채널 멜 스펙트로그램 특성으로 변환
# sampling_rate=16000: 입력 오디오의 샘플링 레이트 명시
# return_tensors="pt": PyTorch 텐서 형식으로 반환
# .input_features: 변환된 멜 스펙트로그램 특성 추출
# .to(device, dtype=torch_dtype): 텐서를 지정된 디바이스와 데이터 타입으로 이동
input_features = processor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features.to(device, dtype=torch_dtype)

# 모델을 사용하여 음성을 텍스트로 변환 (추론 수행)
# input_features: 멜 스펙트로그램 입력
# return_timestamps=True: 단어/세그먼트별 타임스탬프 정보 포함 (더 정확한 결과)
# 반환값: 예측된 토큰 ID 시퀀스
predicted_ids = model.generate(input_features, return_timestamps=True)

# 예측된 토큰 ID를 사람이 읽을 수 있는 텍스트로 디코딩
# batch_decode: 배치 단위로 여러 시퀀스를 한 번에 디코딩
# skip_special_tokens=True: <|startoftranscript|>, <|endoftext|> 등 특수 토큰 제거
# 반환값: 디코딩된 텍스트 리스트
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# 첫 번째 (유일한) 트랜스크립션 결과 출력
# transcription[0]: 배치의 첫 번째 결과 (단일 오디오 파일이므로 인덱스 0)
print(transcription)
