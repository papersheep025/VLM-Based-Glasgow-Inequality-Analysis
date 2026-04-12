# -*- coding: utf-8 -*-
from __future__ import annotations

import os

DEFAULT_MODEL = os.getenv("VLM_MODEL", "qwen3-vl-plus") 
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-540a1528d0c24d50b7ababd5b3e42871")
