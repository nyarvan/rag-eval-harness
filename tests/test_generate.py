from types import SimpleNamespace

import pytest

from rag_eval.data import Document
from rag_eval.generate import (
    SYSTEM_PROMPT,
    Generator,
    OpenAIChat,
    build_user_prompt,
    cache_key,
)

CONTEXTS = (
    Document('Alpha', 'Alpha', 'alpha text'),
    Document('Beta', 'Beta', 'beta text'),
)


class FakeChat:
    """ChatModel that returns a fixed reply and records every call."""

    def __init__(self, reply: str = ' Paris \n', name: str = 'fake'):
        self.name = name
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def make_generator(tmp_path, chat=None):
    return Generator(chat or FakeChat(), tmp_path)


def test_answer_returns_stripped_reply(tmp_path):
    generator = make_generator(tmp_path)

    assert generator.answer('Capital of France?', CONTEXTS) == 'Paris'


def test_user_prompt_lists_contexts_in_order_then_question():
    prompt = build_user_prompt('Why?', CONTEXTS)

    first = prompt.index('Title: Alpha\nText: alpha text')
    second = prompt.index('Title: Beta\nText: beta text')
    assert first < second < prompt.index('Why?')


def test_system_prompt_asks_for_short_context_only_yes_no_answers():
    text = SYSTEM_PROMPT.lower()

    assert 'only' in text and 'context' in text
    assert '"yes"' in text and '"no"' in text


def test_chat_receives_system_and_user_prompts(tmp_path):
    chat = FakeChat()

    make_generator(tmp_path, chat).answer('Why?', CONTEXTS)

    assert chat.calls == [(SYSTEM_PROMPT, build_user_prompt('Why?', CONTEXTS))]


def test_same_inputs_are_answered_from_disk_cache(tmp_path):
    first_chat = FakeChat()
    make_generator(tmp_path, first_chat).answer('Why?', CONTEXTS)
    second_chat = FakeChat(reply='other')

    answer = make_generator(tmp_path, second_chat).answer('Why?', CONTEXTS)

    assert answer == 'Paris'
    assert second_chat.calls == []


def test_other_question_is_not_served_from_cache(tmp_path):
    chat = FakeChat()
    generator = make_generator(tmp_path, chat)

    generator.answer('Why?', CONTEXTS)
    generator.answer('How?', CONTEXTS)

    assert len(chat.calls) == 2


def test_other_contexts_are_not_served_from_cache(tmp_path):
    chat = FakeChat()
    generator = make_generator(tmp_path, chat)

    generator.answer('Why?', CONTEXTS)
    generator.answer('Why?', CONTEXTS[:1])

    assert len(chat.calls) == 2


def test_other_model_name_is_not_served_from_cache(tmp_path):
    make_generator(tmp_path, FakeChat(name='model-a')).answer('Why?', CONTEXTS)
    other = FakeChat(reply='Rome', name='model-b')

    answer = make_generator(tmp_path, other).answer('Why?', CONTEXTS)

    assert answer == 'Rome'


def test_cache_key_changes_with_every_part():
    base = cache_key('model', 'system', 'user')

    assert cache_key('model2', 'system', 'user') != base
    assert cache_key('model', 'system2', 'user') != base
    assert cache_key('model', 'system', 'user2') != base


def test_cache_key_does_not_confuse_field_boundaries():
    assert cache_key('ab', 'c', 'd') != cache_key('a', 'bc', 'd')


def test_cache_key_is_stable_sha256_hex():
    key = cache_key('model', 'system', 'user')

    assert key == cache_key('model', 'system', 'user')
    assert len(key) == 64 and int(key, 16) >= 0


def test_cache_directory_is_created_and_has_no_temp_files(tmp_path):
    cache_dir = tmp_path / 'nested' / 'answers'

    Generator(FakeChat(), cache_dir).answer('Why?', CONTEXTS)

    names = [path.name for path in cache_dir.iterdir()]
    assert len(names) == 1 and names[0].endswith('.json')


def make_openai_client(content, calls):
    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_openai_chat_sends_deterministic_request():
    calls = []
    chat = OpenAIChat(client=make_openai_client('Paris', calls))

    reply = chat.complete('sys', 'usr')

    assert reply == 'Paris'
    assert calls == [{
        'model': 'gpt-4o-mini',
        'temperature': 0.0,
        'max_tokens': 64,
        'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'usr'},
        ],
    }]


def test_openai_chat_name_covers_sampling_settings():
    client = make_openai_client('x', [])

    assert OpenAIChat(client=client).name != OpenAIChat(
        temperature=0.7, client=client
    ).name
    assert OpenAIChat(client=client).name != OpenAIChat(
        'gpt-4o', client=client
    ).name


def test_openai_chat_rejects_empty_completion():
    chat = OpenAIChat(client=make_openai_client(None, []))

    with pytest.raises(RuntimeError):
        chat.complete('sys', 'usr')
