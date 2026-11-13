# Third-party notices

TurboReflect is built on [vLLM](https://github.com/vllm-project/vllm) v0.26.0, licensed under the Apache License 2.0 (see `LICENSE`). It also adapts parts of the following projects.

## RULER

- Source: https://github.com/NVIDIA/RULER
- Used in: `turboreflect/benchmarks/ruler.py`, the task templates of `scripts/data/synthetic/constants.py` and the task parameters of `scripts/synthetic.yaml`, from which the four tasks are generated.
- License: Apache License 2.0, the same license as this repository (see `LICENSE`).

```
Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
```

## LongBench

- Source: https://github.com/THUDM/LongBench
- Used in: `turboreflect/benchmarks/longbench_v2.py`, the zero-shot prompt `prompts/0shot.txt` and the function `extract_answer` of `pred.py`.
- License: MIT License.

```
MIT License

Copyright (c) 2023 THU-KEG & Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
