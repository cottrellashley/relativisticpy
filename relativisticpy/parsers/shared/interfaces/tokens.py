from abc import ABC, abstractmethod


class IToken(ABC):
    @property
    @abstractmethod
    def type(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ITokenProvider(ABC):
    @abstractmethod
    def new_token(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass

    @abstractmethod
    def new_single_operation_token():
        pass

    @abstractmethod
    def new_double_operation_token():
        pass

    @abstractmethod
    def new_tripple_operation_token():
        pass

    @abstractmethod
    def single_match_exists():
        pass

    @abstractmethod
    def double_match_exists():
        pass

    @abstractmethod
    def tripple_match_exists():
        pass

    @abstractmethod
    def singles():
        pass

    @abstractmethod
    def doubles():
        pass

    @abstractmethod
    def tripples():
        pass
