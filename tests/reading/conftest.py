import pytest

from relativisticpy import Workbook


@pytest.fixture
def wb():
    with Workbook() as workbook:
        yield workbook
