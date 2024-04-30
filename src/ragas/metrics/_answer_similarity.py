from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks


logger = logging.getLogger(__name__)


@dataclass
class AnswerSimilarity(MetricWithLLM, MetricWithEmbeddings):
    """
    Scores the semantic similarity of ground truth with generated answer.
    cross encoder score is used to quantify semantic similarity.
    SAS paper: https://arxiv.org/pdf/2108.06130.pdf

    Attributes
    ----------
    name : str
    model_name:
        The model to be used for calculating semantic similarity
        Defaults open-ai-embeddings
        select cross-encoder model for best results
        https://huggingface.co/spaces/mteb/leaderboard
    threshold:
        The threshold if given used to map output to binary
        Default 0.5
    """

    name: str = "answer_similarity"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.ga  # type: ignore
    is_cross_encoder: bool = False
    threshold: t.Optional[float] = None

    def __post_init__(self: t.Self):
        # only for cross encoder
        if isinstance(self.embeddings, HuggingfaceEmbeddings):
            self.is_cross_encoder = True if self.embeddings.is_cross_encoder else False
            self.embeddings.encode_kwargs = {
                **self.embeddings.encode_kwargs,
            }

    async def _ascore(
        self: t.Self, row: t.Dict, callbacks: Callbacks, is_async: bool
    ) -> float:
        assert self.embeddings is not None, "embeddings must be set"

        ground_truth = t.cast(str, row["ground_truth"])
        answer = t.cast(str, row["answer"])

        if self.is_cross_encoder and isinstance(self.embeddings, HuggingfaceEmbeddings):
            raise NotImplementedError(
                "async score [ascore()] not implemented for HuggingFace embeddings"
            )
        else:
            embedding_1 = np.array(await self.embeddings.embed_text(ground_truth))
            embedding_2 = np.array(await self.embeddings.embed_text(answer))
            # Normalization factors of the above embeddings
            norms_1 = np.linalg.norm(embedding_1, keepdims=True)
            norms_2 = np.linalg.norm(embedding_2, keepdims=True)
            embedding_1_normalized = embedding_1 / norms_1
            embedding_2_normalized = embedding_2 / norms_2
            similarity = embedding_1_normalized @ embedding_2_normalized.T
            score = similarity.flatten()

        assert isinstance(score, np.ndarray), "Expects ndarray"
        if self.threshold:
            score = score >= self.threshold

        return score.tolist()[0]


answer_similarity = AnswerSimilarity()
