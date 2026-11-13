# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TurboReflect authors

"""Locality-aware P/D: prefill and decode engines, driven from a process with neither.

Nothing is re-exported: the worker imports this package first and must not load
vLLM or torch before it pins `CUDA_VISIBLE_DEVICES`.
"""
