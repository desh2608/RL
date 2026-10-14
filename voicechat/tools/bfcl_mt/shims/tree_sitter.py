"""Stand-in for tree_sitter: gorilla imports it at module load for Java/JavaScript parsing, which the multi-turn
(Python) path never uses. Any real use raises."""


class Language:
    def __init__(self, *a, **k):
        pass


class Parser:
    def __init__(self, *a, **k):
        pass

    def set_language(self, *a, **k):
        pass

    def parse(self, *a, **k):
        raise RuntimeError("tree_sitter stub: Java/JavaScript parsing is not available in scripts/bfcl_mt")
