import argparse

import pytest

from rag_eval.args import positive_int


def test_positive_int_parses_a_positive_number():
    assert positive_int('5') == 5


@pytest.mark.parametrize('text', ['0', '-5', 'abc', '1.5', ''])
def test_positive_int_rejects_everything_else(text):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(text)
