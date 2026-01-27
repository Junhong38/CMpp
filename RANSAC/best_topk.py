from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class Candidate:
    score: float
    rotation: torch.Tensor      # (3, 3)
    translation: torch.Tensor   # (3,) or (3, 1)
    inliers: torch.Tensor       # (N, M) bool or whatever

class TopKBest:
    def __init__(self, k: int = 5):
        self.k = k
        self.best: List[Candidate] = []

    @torch.no_grad()
    def try_add(self, score, rotation, translation, inliers):
        """
        score: float or scalar tensor
        rotation: (3,3) tensor
        translation: (3,) or (3,1) tensor
        inliers: any tensor (usually bool mask)
        """
        # score를 파이썬 float로 변환
        if torch.is_tensor(score):
            score_val = float(score.detach().item())
        else:
            score_val = float(score)

        cand = Candidate(
            score=score_val,
            rotation=rotation.detach().clone(),
            translation=translation.detach().clone(),
            inliers=inliers.detach().clone() if torch.is_tensor(inliers) else inliers
        )

        # 아직 5개 미만이면 그냥 넣기
        if len(self.best) < self.k:
            self.best.append(cand)
            self.best.sort(key=lambda x: x.score, reverse=True)  # score 큰게 best
            return True

        # 이미 5개 있으면, 최하위(score 가장 작은 것)보다 좋을 때만 교체
        if score_val <= self.best[-1].score:
            return False

        self.best[-1] = cand
        self.best.sort(key=lambda x: x.score, reverse=True)
        return True

    def get_sorted(self) -> List[Candidate]:
        # 항상 score 내림차순 정렬 유지
        return self.best
