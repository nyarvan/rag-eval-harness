"""Answer generation from retrieved contexts, with an on-disk answer cache.

A chat model is anything with a `name` and `complete(system, user)`. The
cache key hashes the model name together with both prompts, so changing the
model, its sampling settings (part of `name`) or either prompt never returns
an old answer.
"""
import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .data import Document

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = 'data/answer_cache'
DEFAULT_MODEL = 'gpt-4o-mini'
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 64

SYSTEM_PROMPT = (
    'Answer the question using only the provided context. '
    'Reply with the shortest possible answer: a few words, not a sentence. '
    'If the question can be answered with yes or no, reply with exactly '
    '"yes" or "no". '
)

FINAL_ANSWER_MARKER = 'Final answer:'
COT_MAX_TOKENS = 512  # room for the reasoning; 64 would cut it off
COT_SYSTEM_PROMPT = (
    'Answer the question using only the provided context. '
    'First think step by step in at most four short sentences: find the '
    'facts you need in the context and combine them. '
    f'Then write the answer on a new line as "{FINAL_ANSWER_MARKER} '
    '<answer>". '
    'The answer itself must be as short as possible: a few words, not a '
    'sentence. '
    'If the question can be answered with yes or no, the answer is exactly '
    '"yes" or "no".'
)
_FINAL_ANSWER = re.compile(r'final answer:\s*(.*)', re.IGNORECASE)


class ChatModel(Protocol):
    """A chat model: `name` identifies the model and its sampling settings."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to the system and user prompts."""
        ...


class OpenAIChat:
    """`ChatModel` over the OpenAI chat completions API.

    The client is built on the first call, from OPENAI_API_KEY, unless one
    is passed in.
    """

    def __init__(
            self, model: str = DEFAULT_MODEL, *,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            client: Any = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client
        self.name = (
            f'openai/{model}?temperature={temperature}&max_tokens={max_tokens}'
        )

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f'{self.name} returned an empty completion')
        return content


def format_context(document: Document) -> str:
    """One context passage, in the same format the retriever embeds."""
    return f'Title: {document.title}\nText: {document.text}'


def build_user_prompt(question: str, contexts: Sequence[Document]) -> str:
    """Contexts in rank order, then the question."""
    passages = '\n\n'.join(format_context(document) for document in contexts)
    return f'Context:\n{passages}\n\nQuestion: {question}\nAnswer:'


def extract_final_answer(reply: str) -> str:
    """The text after the last `Final answer:` marker, up to the line's end.

    A reply without the marker (for example one cut off by the token limit)
    gives an empty answer and a warning, so it is never mistaken for a guess.
    """
    matches = _FINAL_ANSWER.findall(reply)
    if not matches:
        logger.warning('No "%s" in reply: %r', FINAL_ANSWER_MARKER, reply[-80:])
        return ''
    return matches[-1].strip()


def cache_key(model_name: str, system: str, user: str) -> str:
    """SHA-256 of the model name and both prompts, unambiguously encoded."""
    payload = json.dumps(
        [model_name, system, user], ensure_ascii=False, separators=(',', ':')
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class Generator:
    """Answer questions with a chat model, remembering answers on disk.

    With `reasoning` the model thinks first and the answer is taken from its
    `Final answer:` line; the full reply is kept in the cache file.
    """

    def __init__(
            self, chat: ChatModel, cache_dir: str | Path = DEFAULT_CACHE_DIR,
            *, reasoning: bool = False,
    ):
        self._chat = chat
        self._cache_dir = Path(cache_dir)
        self._reasoning = reasoning
        self._system = COT_SYSTEM_PROMPT if reasoning else SYSTEM_PROMPT

    def answer(self, question: str, contexts: Sequence[Document]) -> str:
        """The model's stripped answer, from the cache when it is there."""
        user = build_user_prompt(question, contexts)
        key = cache_key(self._chat.name, self._system, user)
        path = self._cache_dir / f'{key}.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))['answer']
        reply = self._chat.complete(self._system, user)
        answer = (
            extract_final_answer(reply) if self._reasoning else reply.strip()
        )
        self._write(path, user, reply, answer)
        return answer

    def _write(self, path: Path, user: str, reply: str, answer: str) -> None:
        """Store the answer with its inputs; the file appears all at once."""
        record = {
            'model': self._chat.name,
            'system': self._system,
            'user': user,
            'reply': reply,
            'answer': answer,
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temporary, path)
