import time
import random

class MockModelRunner:
    """
    GPU를 쓰지 않고, vLLM 엔진이 내리는 '추론 명령'을 
    가짜(Mock)로 수행하는 클래스입니다.
    """
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size

    def call(self, method, *args):
        """
        nano-vllm의 ModelRunner.call 인터페이스를 흉내냅니다.
        """
        if method == "run":
            return self.run(*args)
        elif method == "exit":
            return None
        return None

    def run(self, seqs, is_prefill):
        """
        실제 Attention 연산 대신, 잠시 대기하고 가짜 토큰 ID를 반환합니다.
        """
        # 1. 연산 시간 시뮬레이션 (Prefill은 토큰이 많으므로 조금 더 오래 걸림)
        if is_prefill:
            # 토큰 개수에 비례하는 연산 시간 (0.01~0.05초)
            num_tokens = sum(len(seq) for seq in seqs)
            time.sleep(min(0.05, num_tokens * 0.0001))
        else:
            # Decode는 한 단어씩 생성하므로 매우 빠름
            time.sleep(0.005)

        # 2. 결과 토큰 생성 (각 요청당 1개의 다음 토큰 ID)
        # 실제로는 모델의 확률값을 구해야 하지만, 여기서는 랜덤 혹은 다음 정수를 줍니다.
        token_ids = []
        for seq in seqs:
            # 간단하게 다음 숫자를 토큰으로 주거나 랜덤 값을 줍니다.
            next_token = (seq.token_ids[-1] + 1) % self.vocab_size
            token_ids.append(next_token)
            
        return token_ids
