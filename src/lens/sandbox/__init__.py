"""Sandbox-side package: passive executor + RLM façade.

This package is copied into the ``lens-repl`` image. It must not import the
host ``lens`` package, hold credentials, or import torch / a DB driver.
"""
