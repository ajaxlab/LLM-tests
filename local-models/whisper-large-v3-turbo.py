# 스크립트가 위치한 디렉토리 경로를 가져오기 위한 모듈
import os

# PyTorch 딥러닝 프레임워크 - 텐서 연산 및 GPU 가속 지원
import torch

# 오디오 파일 로딩 및 처리를 위한 라이브러리 (FFmpeg 의존성 없이 mp3 로드 가능)
import librosa

# JSON 출력을 위한 모듈
import json

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

# 타임스탬프 토큰의 시작 ID (이 ID부터가 타임스탬프 토큰)
# Whisper는 50365를 0.00초로 사용, 이후 0.02초 단위로 증가
timestamp_begin = processor.tokenizer.convert_tokens_to_ids("<|0.00|>")

# 청킹 설정
CHUNK_LENGTH_S = 30  # Whisper 최대 입력 길이 (30초)
STRIDE_LENGTH_S = 5  # 청크 간 오버랩 (경계 부분 정확도 향상)
SAMPLE_RATE = 16000  # Whisper가 요구하는 샘플링 레이트


def format_time(seconds):
    """초 단위를 MM:SS.ss 형식으로 변환"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def parse_tokens_to_segments(token_ids, time_offset=0.0):
    """
    토큰 ID 시퀀스를 타임스탬프가 포함된 세그먼트로 파싱

    Args:
        token_ids: 모델이 생성한 토큰 ID 리스트
        time_offset: 현재 청크의 시작 시간 오프셋 (초)

    Returns:
        segments: [{start, end, text}, ...] 형식의 세그먼트 리스트
        last_end_time: 마지막 세그먼트의 종료 시간
    """
    segments = []
    current_tokens = []
    start_time = None
    last_end_time = time_offset
    internal_offset = 0.0  # 30초 청크 내부 리셋 오프셋
    last_raw_timestamp = 0.0  # 이전 raw 타임스탬프 (리셋 감지용)

    for token_id in token_ids:
        # 타임스탬프 토큰인지 확인 (timestamp_begin 이상이면 타임스탬프)
        if token_id >= timestamp_begin:
            # 타임스탬프 값 계산: (token_id - timestamp_begin) * 0.02초
            raw_timestamp = (token_id - timestamp_begin) * 0.02

            # 타임스탬프가 이전보다 작으면 30초 경계에서 리셋된 것
            if raw_timestamp < last_raw_timestamp - 1.0:
                internal_offset += 30.0

            last_raw_timestamp = raw_timestamp
            timestamp = raw_timestamp + time_offset + internal_offset

            if start_time is None:
                # 첫 타임스탬프 = 세그먼트 시작
                start_time = timestamp
            else:
                # 두 번째 타임스탬프 = 세그먼트 종료
                if current_tokens:
                    text = processor.tokenizer.decode(current_tokens, skip_special_tokens=True)
                    if text.strip():
                        segments.append({
                            "start": format_time(start_time),
                            "end": format_time(timestamp),
                            "text": text.strip()
                        })
                        last_end_time = timestamp
                current_tokens = []
                start_time = timestamp
        else:
            # 일반 토큰은 현재 세그먼트에 추가
            current_tokens.append(token_id)

    return segments, last_end_time


def parse_time_to_seconds(time_str):
    """MM:SS.ss 형식의 시간 문자열을 초 단위로 변환"""
    parts = time_str.split(":")
    return float(parts[0]) * 60 + float(parts[1])


def transcribe_audio(audio_path):
    """
    오디오 파일을 청킹하여 전사 (긴 오디오 지원)

    Args:
        audio_path: 오디오 파일 경로

    Returns:
        segments: 전체 오디오의 타임스탬프가 포함된 세그먼트 리스트
    """
    # librosa로 오디오 파일 로드
    # sr=16000: Whisper 모델이 요구하는 16kHz 샘플링 레이트로 리샘플링
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    # 오디오 총 길이 (초)
    total_duration = len(audio) / SAMPLE_RATE
    print(f"오디오 길이: {format_time(total_duration)}")

    # 청크 및 스트라이드 샘플 수 계산
    chunk_samples = CHUNK_LENGTH_S * SAMPLE_RATE
    stride_samples = STRIDE_LENGTH_S * SAMPLE_RATE

    all_segments = []
    chunk_start = 0
    chunk_index = 0
    last_end_time = 0.0  # 이전 청크의 마지막 세그먼트 종료 시간

    while chunk_start < len(audio):
        chunk_index += 1
        # 현재 청크 추출
        chunk_end = min(chunk_start + chunk_samples, len(audio))
        audio_chunk = audio[chunk_start:chunk_end]

        # 청크 시작 시간 (초)
        time_offset = chunk_start / SAMPLE_RATE
        print(f"청크 {chunk_index} 처리 중: {format_time(time_offset)} ~ {format_time(chunk_end / SAMPLE_RATE)}")

        # 오디오 데이터를 모델 입력 형식으로 변환
        input_features = processor(
            audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device, dtype=torch_dtype)

        # 모델 추론
        predicted_ids = model.generate(input_features, return_timestamps=True)

        # 토큰을 세그먼트로 파싱
        token_ids = predicted_ids[0].tolist()
        segments, _ = parse_tokens_to_segments(token_ids, time_offset)

        # 오버랩 영역 중복 제거: 이전 청크의 마지막 종료 시간 이후 세그먼트만 포함
        if all_segments and segments:
            filtered_segments = []
            for seg in segments:
                seg_start = parse_time_to_seconds(seg["start"])
                # 이전 청크 종료 시간 이후에 시작하는 세그먼트만 추가
                if seg_start >= last_end_time - 0.5:  # 0.5초 여유
                    filtered_segments.append(seg)
            segments = filtered_segments

        # 현재 청크의 세그먼트 추가
        if segments:
            all_segments.extend(segments)
            # 마지막 세그먼트의 종료 시간 업데이트
            last_end_time = parse_time_to_seconds(segments[-1]["end"])

        # 다음 청크 시작 위치 (스트라이드만큼 이동)
        chunk_start += chunk_samples - stride_samples

        # 마지막 청크면 종료
        if chunk_end >= len(audio):
            break

    return all_segments


# 메인 실행
if __name__ == "__main__":
    # 이 스크립트 파일이 위치한 디렉토리에서 sample.mp3 파일의 전체 경로 생성
    sample = os.path.join(os.path.dirname(__file__), "sample.mp3")

    # 오디오 전사 실행
    segments = transcribe_audio(sample)

    # JSON 형식으로 보기 좋게 출력
    print("\n=== 전사 결과 ===")
    print(json.dumps(segments, ensure_ascii=False, indent=2))
