import os
import argparse
from huggingface_hub import snapshot_download

def download_hf_model(repo_id, local_dir):
    """
    HuggingFace 모델을 로컬 디렉토리에 다운로드합니다.
    """
    # 1. 경로 설정
    # repo_id의 마지막 부분(예: deepseek-coder-7b-instruct-v1.5)을 폴더명으로 사용
    model_name = repo_id.split("/")[-1]
    save_path = os.path.join(local_dir, model_name)
    
    print(f"🚀 다운로드 시작: {repo_id}")
    print(f"📁 저장 위치: {save_path}")
    
    # 2. 다운로드 실행 (이미 있는 파일은 건너뛰고, 부족한 파일만 받음)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            local_dir_use_symlinks=False, # 윈도우에서는 symlink 대신 직접 복사가 안전함
            token=None # 필요한 경우 HF 토큰 입력
        )
        print(f"✅ 다운로드 완료! 이제 다음 명령어로 모델을 분석해 보세요.")
        print(f"👉 python src/inspect_model.py {save_path}")
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace 모델 Repo ID (예: deepseek-ai/deepseek-coder-7b-instruct-v1.5)")
    parser.add_argument("--dest", type=str, default="models", help="모델을 저장할 상위 디렉토리 (기본값: models)")
    args = parser.parse_args()
    
    # 디렉토리 생성
    dest_path = os.path.expanduser(args.dest)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        
    download_hf_model(args.repo, dest_path)

