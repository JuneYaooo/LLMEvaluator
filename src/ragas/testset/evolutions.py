from __future__ import annotations

import logging
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from src.ragas.exceptions import MaxRetriesExceeded
from src.ragas.llms import BaseRagasLLM
from src.ragas.llms.json_load import json_loader
from src.ragas.llms.prompt import Prompt
from src.ragas.run_config import RunConfig
from src.ragas.testset.docstore import DocumentStore, Node
from src.ragas.testset.filters import EvolutionFilter, NodeFilter, QuestionFilter
from src.ragas.testset.prompts import (
    compress_question_prompt,
    conditional_question_prompt,
    find_relevant_context_prompt,
    multi_context_question_prompt,
    question_answer_prompt,
    question_rewrite_prompt,
    reasoning_question_prompt,
    seed_question_prompt,
    counterfactual_prompt,
    error_correction_prompt
)
from src.ragas.testset.utils import rng

logger = logging.getLogger(__name__)


@dataclass
class CurrentNodes:
    root_node: Node
    nodes: t.List[Node] = field(default_factory=list)


# (question, current_nodes, evolution_type)
EvolutionOutput = t.Tuple[str, CurrentNodes, str]


class DataRow(BaseModel):
    question: str
    question_raw: str
    contexts: t.List[str]
    contexts_raw: t.List[str]
    ground_truth: t.Union[str, float] = np.nan
    evolution_type: str
    metadata: t.List[dict]


@dataclass
class Evolution:
    generator_llm: BaseRagasLLM = t.cast(BaseRagasLLM, None)
    docstore: t.Optional[DocumentStore] = None
    node_filter: t.Optional[NodeFilter] = None
    question_filter: t.Optional[QuestionFilter] = None
    question_answer_prompt: Prompt = field(
        default_factory=lambda: question_answer_prompt
    )
    find_relevant_context_prompt: Prompt = field(
        default_factory=lambda: find_relevant_context_prompt
    )
    rewrite_invalid_question_prompt: Prompt = field(
        default_factory=lambda: question_rewrite_prompt
    )
    max_tries: int = 5
    is_async: bool = True

    @staticmethod
    def merge_nodes(nodes: CurrentNodes) -> Node:
        # TODO: while merging merge according to the order of documents
        # if any nodes from same document take account their page order

        new_node = Node(
            doc_id="merged",
            page_content="\n".join(n.page_content for n in nodes.nodes),
            keyphrases=[phrase for n in nodes.nodes for phrase in n.keyphrases],
        )

        embed_dim = (
            len(nodes.nodes[0].embedding)
            if nodes.nodes[0].embedding is not None
            else None
        )
        if embed_dim:
            node_embeddings = np.array([n.embedding for n in nodes.nodes]).reshape(
                -1, embed_dim
            )
            new_node.embedding = np.average(node_embeddings, axis=0)
        return new_node

    def init(self, is_async: bool = True, run_config: t.Optional[RunConfig] = None):
        self.is_async = is_async
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def set_run_config(self, run_config: RunConfig):
        if self.docstore:
            self.docstore.set_run_config(run_config)
        if self.generator_llm:
            self.generator_llm.set_run_config(run_config)
        if self.node_filter:
            self.node_filter.set_run_config(run_config)
        if self.question_filter:
            self.question_filter.set_run_config(run_config)

    async def aretry_evolve(
        self,
        current_tries: int,
        current_nodes: CurrentNodes,
        update_count: bool = True,
    ) -> EvolutionOutput:
        if update_count:
            current_tries += 1
        logger.info("retrying evolution: %s times", current_tries)
        if current_tries > self.max_tries:
            # TODO: make this into a custom exception
            raise MaxRetriesExceeded(self)
        return await self._aevolve(current_tries, current_nodes)

    async def _transform_question(self, prompt: Prompt, question: str) -> str:
        assert self.generator_llm is not None, "generator_llm cannot be None"

        results = await self.generator_llm.generate(
            prompt=prompt.format(question=question), is_async=self.is_async
        )
        return results.generations[0][0].text.strip()

    def _get_new_random_node(self):
        assert self.docstore is not None, "docstore cannot be None"
        new_node = self.docstore.get_random_nodes(k=1)[0]
        return CurrentNodes(root_node=new_node, nodes=[new_node])

    async def evolve(self, current_nodes: CurrentNodes) -> DataRow:
        # init tries with 0 when first called
        current_tries = 0
        (
            evolved_question,
            current_nodes,
            evolution_type,
            question_raw,
            current_nodes_raw,
        ) = await self._aevolve(current_tries, current_nodes)

        return await self.generate_datarow(
            question=evolved_question,
            current_nodes=current_nodes,
            evolution_type=evolution_type,
            question_raw=question_raw,
            current_nodes_raw=current_nodes_raw
        )

    async def fix_invalid_question(
        self, question: str, current_nodes: CurrentNodes, feedback: str
    ):
        """
        if the question is invalid, get more nodes and retry
        """
        prev_node = current_nodes.root_node.prev
        if prev_node is not None:
            current_nodes.nodes.insert(0, prev_node)
            current_nodes.root_node = prev_node
            prompt = self.rewrite_invalid_question_prompt.format(
                question=question,
                context=self.merge_nodes(current_nodes).page_content,
                feedback=feedback,
            )
            results = await self.generator_llm.generate(
                prompt=prompt, is_async=self.is_async
            )
            question = results.generations[0][0].text.strip()

        return question, current_nodes

    @abstractmethod
    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        ...

    async def filter_and_retry(self, question):
        ...

    async def generate_datarow(
        self,
        question: str,
        current_nodes: CurrentNodes,
        evolution_type: str,
        question_raw: str = "",
        current_nodes_raw: t.Optional[t.List[Node]] = None,
    ):
        assert self.generator_llm is not None, "generator_llm cannot be None"

        node_content = [
            f"{i+1}\t{n.page_content}" for i, n in enumerate(current_nodes.nodes)
        ]
        results = await self.generator_llm.generate(
            prompt=self.find_relevant_context_prompt.format(
                question=question, contexts=node_content
            )
        )
        relevant_contexts_result = await json_loader.safe_load(
            results.generations[0][0].text.strip(), llm=self.generator_llm
        )
        relevant_context_indices = (
            relevant_contexts_result.get("relevant_contexts", None)
            if isinstance(relevant_contexts_result, dict)
            else None
        )
        if relevant_context_indices is None:
            relevant_context = CurrentNodes(
                root_node=current_nodes.root_node, nodes=current_nodes.nodes
            )
        else:
            selected_nodes = [
                current_nodes.nodes[i - 1]
                for i in relevant_context_indices
                if i - 1 < len(current_nodes.nodes)
            ]
            relevant_context = (
                CurrentNodes(root_node=selected_nodes[0], nodes=selected_nodes)
                if selected_nodes
                else current_nodes
            )

        merged_nodes = self.merge_nodes(relevant_context)
        results = await self.generator_llm.generate(
            prompt=self.question_answer_prompt.format(
                question=question, context=merged_nodes.page_content
            )
        )
        answer = await json_loader.safe_load(
            results.generations[0][0].text.strip(), self.generator_llm
        )
        answer = answer if isinstance(answer, dict) else {}
        logger.debug("answer generated: %s", answer)
        answer = (
            np.nan if answer.get("verdict") == "-1" else answer.get("answer", np.nan)
        )

        return DataRow(
            question=question.strip('"'),
            question_raw=question_raw.strip('"'),
            contexts=[n.page_content for n in relevant_context.nodes],
            contexts_raw = current_nodes_raw,
            ground_truth=answer,
            evolution_type=evolution_type,
            metadata=[n.metadata for n in relevant_context.nodes],
        )

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        """
        Adapt the filter to a different language.
        """
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        self.question_answer_prompt = self.question_answer_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.find_relevant_context_prompt = self.find_relevant_context_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.rewrite_invalid_question_prompt = (
            self.rewrite_invalid_question_prompt.adapt(
                language, self.generator_llm, cache_dir
            )
        )
        self.node_filter.adapt(language, cache_dir)
        self.question_filter.adapt(language, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        """
        Save the filter prompts to a path.
        """
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        self.question_answer_prompt.save(cache_dir)
        self.find_relevant_context_prompt.save(cache_dir)
        self.node_filter.save(cache_dir)
        self.question_filter.save(cache_dir)


@dataclass
class SimpleEvolution(Evolution):
    seed_question_prompt: Prompt = field(default_factory=lambda: seed_question_prompt)

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.filter(merged_node)
        if not passed["score"]:
            current_nodes = self._get_new_random_node()
            return await self.aretry_evolve(
                current_tries, current_nodes, update_count=False
            )

        logger.debug("keyphrases in merged node: %s", merged_node.keyphrases)
        prompt=self.seed_question_prompt.format(
                context=merged_node.page_content,
                keyphrase=rng.choice(np.array(merged_node.keyphrases), size=1)[0],
            )
        results = await self.generator_llm.generate(
            prompt=prompt
        )
        seed_question = results.generations[0][0].text
        logger.info("seed question generated: %s", seed_question)
        is_valid_question, feedback = await self.question_filter.filter(seed_question)

        if not is_valid_question:
            # get more context to rewrite question
            seed_question, current_nodes = await self.fix_invalid_question(
                seed_question, current_nodes, feedback
            )
            logger.info("rewritten question: %s", seed_question)
            is_valid_question, _ = await self.question_filter.filter(seed_question)
            if not is_valid_question:
                # retry with new nodes added
                current_nodes = self._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        return seed_question, current_nodes, "simple", "", []

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.seed_question_prompt = self.seed_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.seed_question_prompt.save(cache_dir)


@dataclass
class ComplexEvolution(Evolution):
    se: t.Optional[SimpleEvolution] = field(default=None, repr=False)
    evolution_filter: t.Optional[EvolutionFilter] = field(default=None, repr=False)
    compress_question_prompt: Prompt = field(
        default_factory=lambda: compress_question_prompt
    )

    def init(self, is_async: bool = True, run_config: t.Optional[RunConfig] = None):
        if run_config is None:
            run_config = RunConfig()
        super().init(is_async=is_async, run_config=run_config)

        if self.se is None:
            # init simple evolution to get seed question
            self.se = SimpleEvolution(
                generator_llm=self.generator_llm,
                docstore=self.docstore,
                node_filter=self.node_filter,
                question_filter=self.question_filter,
            )
        # init evolution filter with critic llm from another filter
        assert self.node_filter is not None, "node filter cannot be None"
        if self.evolution_filter is None:
            self.evolution_filter = EvolutionFilter(self.node_filter.llm)

        # set run configs
        self.se.set_run_config(run_config)
        self.evolution_filter.set_run_config(run_config)

    async def _acomplex_evolution(
        self, current_tries: int, current_nodes: CurrentNodes, question_prompt: Prompt
    ):
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, current_nodes, _, ax, bx = await self.se._aevolve(
            current_tries, current_nodes
        )
        logger.debug(
            "[%s] simple question generated: %s",
            self.__class__.__name__,
            simple_question,
        )

        merged_node = self.merge_nodes(current_nodes)
        result = await self.generator_llm.generate(
            prompt=question_prompt.format(
                question=simple_question, context=merged_node.page_content
            )
        )
        reasoning_question = result.generations[0][0].text.strip()
        is_valid_question, feedback = await self.question_filter.filter(
            reasoning_question
        )
        if not is_valid_question:
            # retry
            reasoning_question, current_nodes = await self.fix_invalid_question(
                reasoning_question, current_nodes, feedback
            )
            logger.info("rewritten question: %s", reasoning_question)
            is_valid_question, _ = await self.question_filter.filter(reasoning_question)
            if not is_valid_question:
                # retry with new nodes added
                current_nodes = self.se._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        # compress the question
        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=reasoning_question
        )
        logger.debug(
            "[%s] question compressed: %s",
            self.__class__.__name__,
            reasoning_question,
        )

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if await self.evolution_filter.filter(simple_question, compressed_question):
            # retry
            current_nodes = self.se._get_new_random_node()
            logger.debug(
                "evolution_filter failed, retrying with %s", len(current_nodes.nodes)
            )
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes, simple_question, ""

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.evolution_filter is not None, "evolution filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        super().adapt(language, cache_dir)
        self.se.adapt(language, cache_dir)
        self.compress_question_prompt = self.compress_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )
        self.evolution_filter.adapt(language, cache_dir)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        assert self.evolution_filter is not None, "evolution filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        super().save(cache_dir)
        self.se.save(cache_dir)
        self.evolution_filter.save(cache_dir)
        self.compress_question_prompt.save(cache_dir)


@dataclass
class MultiContextEvolution(ComplexEvolution):
    multi_context_question_prompt: Prompt = field(
        default_factory=lambda: multi_context_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, current_nodes, _, ax, bx = await self.se._aevolve(
            current_tries, current_nodes
        )
        logger.debug(
            "[MultiContextEvolution] simple question generated: %s", simple_question
        )
        # find a similar node and generate a question based on both
        merged_node = self.merge_nodes(current_nodes)
        similar_node = self.docstore.get_similar(merged_node, top_k=1)
        if not similar_node:
            # retry
            new_random_nodes = self.docstore.get_random_nodes(k=1)
            current_nodes = CurrentNodes(
                root_node=new_random_nodes[0], nodes=new_random_nodes
            )
            return await self.aretry_evolve(current_tries, current_nodes)
        else:
            assert isinstance(similar_node[0], Node), "similar_node must be a Node"
            current_nodes.nodes.append(similar_node[0])

        prompt = self.multi_context_question_prompt.format(
            question=simple_question,
            context1=merged_node.page_content,
            context2=similar_node[0].page_content,
        )
        results = await self.generator_llm.generate(prompt=prompt)
        question = results.generations[0][0].text.strip()
        logger.debug(
            "[MultiContextEvolution] multicontext question generated: %s", question
        )
        is_valid_question, feedback = await self.question_filter.filter(question)
        if not is_valid_question:
            # retry
            # get more context to rewrite question
            question, current_nodes = await self.fix_invalid_question(
                question, current_nodes, feedback
            )
            logger.info("rewritten question: %s", question)
            is_valid_question, _ = await self.question_filter.filter(question)

            if not is_valid_question:
                # retry with new nodes added
                current_nodes = self.se._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        # compress the question
        compressed_question = await self._transform_question(
            prompt=self.compress_question_prompt, question=question
        )
        logger.debug(
            "[MultiContextEvolution] multicontext question compressed: %s",
            compressed_question,
        )

        assert self.evolution_filter is not None, "evolution filter cannot be None"
        if await self.evolution_filter.filter(simple_question, compressed_question):
            # retry
            current_nodes = self.se._get_new_random_node()
            return await self.aretry_evolve(current_tries, current_nodes)

        return compressed_question, current_nodes, "multi_context","",[]

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.multi_context_question_prompt = self.multi_context_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.multi_context_question_prompt.save(cache_dir)


@dataclass
class ReasoningEvolution(ComplexEvolution):
    reasoning_question_prompt: Prompt = field(
        default_factory=lambda: reasoning_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        current_nodes = self.docstore.get_max_context(current_nodes.nodes[0], 5000)
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.reasoning_question_prompt
        )
        return result[0], result[1], "reasoning", result[2], []

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.reasoning_question_prompt = self.reasoning_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.reasoning_question_prompt.save(cache_dir)


@dataclass
class ConditionalEvolution(ComplexEvolution):
    conditional_question_prompt: Prompt = field(
        default_factory=lambda: conditional_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        result = await self._acomplex_evolution(
            current_tries, current_nodes, self.conditional_question_prompt
        )
        return result[0], result[1], "conditional","",[]

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.conditional_question_prompt = self.conditional_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.conditional_question_prompt.save(cache_dir)


# After the generation is completed, change the document into a counterfactual document
@dataclass
class CounterfactualEvolution(ComplexEvolution):
    reasoning_question_prompt: Prompt = field(
        default_factory=lambda: reasoning_question_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, current_nodes, _, ax, bx = await self.se._aevolve(
            current_tries, current_nodes
        )
        logger.debug(
            "[%s] simple question generated: %s",
            self.__class__.__name__,
            simple_question,
        )

        merged_node = self.merge_nodes(current_nodes)
        prompt = counterfactual_prompt.format(
                question=simple_question, context=merged_node.page_content)
        result = await self.generator_llm.generate(
            prompt=prompt
        )
        counterfactual_content = result.generations[0][0].text.strip()

        return simple_question, current_nodes, "counter_factual", "", [counterfactual_content]
        
    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.reasoning_question_prompt = self.reasoning_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.reasoning_question_prompt.save(cache_dir)

@dataclass
class ErrorCorrectionEvolution(ComplexEvolution):
    error_correction_prompt: Prompt = field(
        default_factory=lambda: error_correction_prompt
    )

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"
        assert self.se is not None, "simple evolution cannot be None"

        simple_question, current_nodes, _, ax, bx = await self.se._aevolve(
            current_tries, current_nodes
        )
        logger.debug(
            "[%s] simple question generated: %s",
            self.__class__.__name__,
            simple_question,
        )

        merged_node = self.merge_nodes(current_nodes)
        result = await self.generator_llm.generate(
            prompt=error_correction_prompt.format(
                question=simple_question,
            )
        )
        error_question = result.generations[0][0].text.strip()

        return error_question, current_nodes,"error_correction", simple_question, []

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.error_correction_prompt = self.error_correction_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        super().save(cache_dir)
        self.error_correction_prompt.save(cache_dir)


simple = SimpleEvolution()
multi_context = MultiContextEvolution()
reasoning = ReasoningEvolution()
conditional = ConditionalEvolution()
counterfactual = CounterfactualEvolution()
error_correction = ErrorCorrectionEvolution()