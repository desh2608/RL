"""Minimal stand-in for the `overrides` package (only its decorators are used by gorilla bfcl_eval)."""


def final(f):
    return f


def override(f):
    return f


class EnforceOverrides:
    pass
