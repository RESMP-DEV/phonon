from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._]*"
SPOKEN_SEPARATOR_PATTERN = r"underscore\s+underscore|double\s+underscore|underscore|dash|hyphen"

SPOKEN_SEPARATOR_RUN_RE = re.compile(
    rf"\b{TOKEN_PATTERN}\b(?:\s+(?:{SPOKEN_SEPARATOR_PATTERN})\s+\b{TOKEN_PATTERN}\b)+",
    flags=re.IGNORECASE,
)
SPOKEN_SEPARATOR_SPLIT_RE = re.compile(
    rf"\s+({SPOKEN_SEPARATOR_PATTERN})\s+",
    flags=re.IGNORECASE,
)

SMALL_INTEGER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
DIMENSION_CONTEXT_RE = re.compile(
    r"\b(?:conv(?:olution)?|kernel|filter|stride|matrix|tensor|tile|grid|window|"
    r"shape|dimension|dimensions|resolution|image|feature\s+map)\b",
    flags=re.IGNORECASE,
)
DIMENSION_BY_RE = re.compile(
    r"\b(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s+by\s+"
    r"(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ExactFormRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str
    category: str


def _rule(name: str, pattern: str, replacement: str, category: str) -> ExactFormRule:
    return ExactFormRule(
        name=name,
        pattern=re.compile(pattern, flags=re.IGNORECASE),
        replacement=replacement,
        category=category,
    )


RULES: tuple[ExactFormRule, ...] = (
    _rule(
        "flops_utilization",
        r"\bflops?\s+(?:equalization|sequalization|utilization)\b",
        "FLOPs utilization",
        "gpu_metric",
    ),
    _rule("freecodecamp", r"\bfree\s+code\s*camp\b", "FreeCodeCamp", "brand"),
    _rule("elliot_arledge", r"\belliot\s+alridge\b", "Elliot Arledge", "name"),
    _rule("hugging_phase", r"\bhugging\s+phase\b", "Hugging Face", "brand"),
    _rule("google_cloud", r"\b(?:google\s+color|global\s+color)\b", "Google Cloud", "cloud"),
    _rule("inference_time", r"\binterference\s+time\b", "inference time", "ml_term"),
    _rule("llm_serving", r"\bell\s*em\s+serving\b", "LLM serving", "ml_term"),
    _rule("mnist", r"\bm[\s-]*nest\b", "MNIST", "dataset"),
    _rule("perceptron", r"\bmultilayer\s+perception\b", "multilayer perceptron", "ml_term"),
    _rule("matmuls", r"\bmap\s*muls?\b|\bmapmules\b|\bmapmuls\b", "matmuls", "math"),
    _rule("matmul", r"\bmat\s+(?:mole|mall)\b|\bmatmall\b", "matmul", "math"),
    _rule("ceiling_function", r"\bsealing\s+function\b", "ceiling function", "math"),
    _rule("cxx_weird", r"\bC\s+C\s+[\ufffd\u2047?](?=\s|$)|\bC\s+plus\s+plus\b|\bC\s*\?\?(?=\s|$)", "C++", "language"),
    _rule("cuda_cxx", r"\bcuda\s+C\s+plus\s+plus\b", "CUDA C++", "language"),
    _rule("hugging_face", r"\bhuggin\s*face\b|\bhuggingface\b", "Hugging Face", "brand"),
    _rule("pytorch", r"\bpi\s*torch\b|\bpy\s*torch\b", "PyTorch", "framework"),
    _rule("pytorch_mangled", r"\bpydorch\b|\bpypoint\b|\bpytras\b", "PyTorch", "framework"),
    _rule("torchvision", r"\btorfee\b", "Torch Vision", "package"),
    _rule("torchvision_install", r"\btorch\s+vision\b(?=\s+torch\s+audio\s+pytorch\s+cuda\b)", "torchvision", "package"),
    _rule("torchaudio_install", r"\btorch\s+audio\b(?=\s+pytorch\s+cuda\b)", "torchaudio", "package"),
    _rule("pytorch_cuda_eq", r"\bpytorch\s+cuda\s+(?:is\s+)?equal\s+to\s+([0-9.]+)\b", r"pytorch-cuda=\1", "package"),
    _rule("for_i_range", r"\b4\s*([ijk])\s+in\s+range\b", r"for \1 in range", "code"),
    _rule("dtype", r"\bd\s*type\s+equals\b", "dtype =", "code"),
    _rule("tl_float32", r"\btl\s*(?:dot|\.)\s*flow\s*32\b|\btl\s*(?:dot|\.)\s*float\s*32\b", "tl.float32", "code"),
    _rule("float32", r"\bflow\s*32\b", "float32", "dtype"),
    _rule("blockidx_x", r"\bblock\s+(?:idx|id\s*x)\s*(?:dot|\.)?\s*x\b", "blockIdx.x", "cuda"),
    _rule("blockdim_x", r"\bblock\s+dim\s*(?:dot|\.)?\s*x\b", "blockDim.x", "cuda"),
    _rule("threadidx_x", r"\bthread\s+(?:idx|id\s*x)\s*(?:dot|\.)?\s*x\b", "threadIdx.x", "cuda"),
    _rule("grid_dim", r"\bgrid\s+dim\b", "gridDim", "cuda"),
    _rule("block_idx", r"\bblock\s+(?:idx|id\s*x)\b", "blockIdx", "cuda"),
    _rule("block_dim", r"\bblock\s+dim\b", "blockDim", "cuda"),
    _rule("thread_idx", r"\bthread\s+(?:idx|id\s*x)\b", "threadIdx", "cuda"),
    _rule("cuda_runtime", r"\bcuda\s+underscore\s+runtime\s+(?:dot|\.)\s*h\b", "cuda_runtime.h", "cuda"),
    _rule("dunder_global", r"\bunderscore\W+underscore\W+global\W+underscore\W+underscore\b", "__global__", "cuda"),
    _rule("dunder_syncthreads", r"\bunderscore\W+underscore\W+sync\s*threads\W+underscore\W+underscore\b|\bsync\s+threads\b", "__syncthreads", "cuda"),
    _rule("dunder_syncwarp", r"\bunderscore\W+underscore\W+sync\s*warp\W+underscore\W+underscore\b|\bsync\s+warps?\b", "__syncwarp", "cuda"),
    _rule("const_float_ptr", r"\bconst\s+float\s+asterisk\b", "const float*", "cuda"),
    _rule("float_ptr", r"\bfloat\s+asterisk\b", "float*", "cuda"),
    _rule("iostream", r"\bi\s*a\s*stream\b", "iostream", "cpp"),
    _rule("vector_add_cu", r"\bvector\s+underscore\s+add\s+(?:dot|\.)\s*c\s*u\b", "vector_add.cu", "cuda"),
    _rule("cublaslt", r"\b(?:cu\s*blas|cublas|kublas|kublos|kublast|cublos)\s*l\s*t\b", "cuBLASLt", "cuda"),
    _rule("cublas", r"\bcublos\b|\bkublas\b|\bkublos\b|\bkubloss\b|\bkublus\b", "cuBLAS", "cuda"),
    _rule("cudnn", r"\bqdnn\b|\bkutynn\b|\bcudnn\b", "cuDNN", "cuda"),
    _rule("sgemm", r"\bsgm(?=\s+cuda\b)|\bsgm00\b|\bsgem\b|\bsgemm\b", "SGEMM", "cuda"),
    _rule("gemm", r"\b(?:implicit|precomp)\s+gem\b|\bgem(?=\s+sgem\b)", "GEMM", "cuda"),
    _rule("gflops_per_second", r"\bg\s*flops?\s+per\s+second\b|\bgigaflops\s+per\s+second\b", "GFLOP/s", "gpu_metric"),
    _rule("gb_per_second", r"\bgigabytes?\s+per\s+second\b|\bgb\s+per\s+second\b", "GB/s", "gpu_metric"),
    _rule("tb_per_second", r"\bterabytes?\s+per\s+second\b|\btb\s+per\s+second\b", "TB/s", "gpu_metric"),
    _rule("fp32", r"\bep32\b|\bfp\s*32\b", "FP32", "dtype"),
    _rule("float4", r"\bfloat\s+four\b", "float4", "cuda"),
    _rule("softmax", r"\bsouthmax\b", "softmax", "ml_op"),
    _rule("openai", r"\bopen\s+ai\b", "OpenAI", "brand"),
    _rule("tensor_rt_llm", r"\btensor\s*rt\s*-?\s*l\s*l\s*m\b|\btensorrt\s*-?\s*l\s*l\s*m\b", "TensorRT-LLM", "model"),
    _rule("trt_llm", r"\btrt\s*-?\s*l\s*l\s*m\b", "TRT-LLM", "model"),
    _rule("triton_inference_server", r"\btryton\s+inference\s+server\b", "Triton inference server", "serving"),
    _rule("triton_next_power_two", r"\btriton\s+dot\s+next\s+power\s+of\s+two\b", "triton.next_power_of_2", "api"),
    _rule("triton_jit", r"\btriton\s+(?:dot|\.)\s*jit\b", "triton.jit", "api"),
    _rule("high_throughput", r"\bhard\s+throughput\s+batch\s+inference\b", "high throughput batch inference", "serving"),
    _rule("turnkey", r"\b10\s+key\s+solution\b", "turnkey solution", "phrase"),
    _rule("ipex", r"\bipax\b|\bi\s*pax\b", "IPEX", "framework"),
    _rule("quantize", r"\bquank'?s\b|\bquanks\b", "quantize", "ml_term"),
    _rule("int8_model", r"\bto\s+int\s*8\s+(?:a\s+)?model\b", "to Int8 model", "dtype"),
    _rule("llm_generate", r"\bllm\s*(?:dot|\.)\s*generate\b", "llm.generate", "api"),
    _rule("from_pretrained", r"\bfrom\s*pretain(?:ed)?\b|\bfrompretain(?:ed)?\b", "from_pretrained", "api"),
    _rule("save_pretrained", r"\bsave\s*pretain(?:ed)?\b|\bsavepretain(?:ed)?\b", "save_pretrained", "api"),
    _rule("autotokenizer", r"\bauto\s+tokenizer\b", "AutoTokenizer", "api"),
    _rule("vision_transformer", r"\bvision\s+transformer\b(?=,?\s+to\s+the\s+from|\s+class\b|\s+model\b)", "VisionTransformer", "api"),
    _rule("transformers_library", r"\btransforms\s+library\b", "Transformers library", "package"),
    _rule("as_array", r"\bas\s+array\b|\basarray\b", "as_array", "api"),
    _rule("imageops", r"\bimage\s+ops\b", "ImageOps", "api"),
    _rule("vllm", r"\bv\s+l\s+l\s+m\b", "vLLM", "framework"),
    _rule("vllm_core", r"\bvom\s+core\b", "vLLM core", "framework"),
    _rule("fastapi", r"\bfast\s+api\b", "FastAPI", "framework"),
    _rule("fastap", r"\bfastap\b", "FastAPI", "framework"),
    _rule("uvicorn", r"\byovicorn\b|\buvi\s*corn\b", "Uvicorn", "framework"),
    _rule("llama3", r"\blama\s+3\b|\blama\s+three\b|\bllama\s+three\b", "Llama 3", "model"),
    _rule("llama_cpp", r"\blama\s+cpp\b|\bllama\s+cpp\b", "llama.cpp", "tool"),
    _rule("meta_llama3", r"\bmeta\s+slash\s+llama\s+3\b", "meta/Llama 3", "model"),
    _rule("deepseek_r1", r"\bdeep\s*(?:sea|seek|c)\s*(?:car|r)\s*1\b|\bdeepcarr1\b", "DeepSeek R1", "model"),
    _rule("ollama", r"\bolama\b|\bor\s+llama\b", "Ollama", "tool"),
    _rule("mixtral_8x7b", r"\bmix(?:ed|tral|rel)?\s*(?:role|roll|rolled|relate)?\s*8\s*(?:x|cross)\s*7\s*b\b", "Mixtral 8x7B", "model"),
    _rule("lm_studio", r"\b(?:alama|llm|lm)\s+studio\b", "LM Studio", "tool"),
    _rule("kv_cache", r"\bkv\s+cash\b", "KV cache", "llm_infra"),
    _rule("qwen", r"\bquen\b", "Qwen", "model"),
    _rule("dlpack", r"\bdl\s*pac\b|\bdlpack\b", "DLPack", "protocol"),
    _rule("cupy", r"\b(?:coupy|kupy)\b", "CuPy", "package"),
    _rule("dockerfile", r"\bdocker\s+file\b", "Dockerfile", "file"),
    _rule("nvfp4", r"\bnvfp\s+four\b", "NVFP4", "dtype"),
    _rule("fp4", r"\bfp\s+four\b", "FP4", "dtype"),
    _rule("h100s", r"\bh\s*10s\b", "H100s", "gpu"),
    _rule("a100s", r"\ba\s+hundreds\b", "A100s", "gpu"),
    _rule("nvcc_version", r"\bnbcc(?:ci)?[- ]?version\b", "nvcc --version", "command"),
    _rule("nvcc", r"\bnbcc\b", "nvcc", "command"),
    _rule("nvidia_smi", r"\bnvidia\s+smi\b", "nvidia-smi", "command"),
    _rule("cuda_malloc", r"\bcuda\s+ma?ll?oc\b|\bcuda\s+malloc\b", "cudaMalloc", "cuda"),
    _rule("cuda_malloc_mangled", r"\bcut\s+a?malloc\b|\bcuda\s*ma?loc\b|\bcuda\s*macl\b", "cudaMalloc", "cuda"),
    _rule("cuda_memcpy_h2d", r"\bcuda\s*mem\s*(?:copy|cpy)\s+host\s+to\s+device\b", "cudaMemcpyHostToDevice", "cuda"),
    _rule("cuda_memcpy_d2h", r"\bcuda\s*mem\s*(?:copy|cpy)\s+device\s+to\s+host\b", "cudaMemcpyDeviceToHost", "cuda"),
    _rule("cuda_memcopy_host_device", r"\bcuda\s*mem\s*(?:copy|cpy)\s+host\s+device\b", "cudaMemcpyHostToDevice", "cuda"),
    _rule("cuda_memcpy_async", r"\bcuda\s*mem\s*(?:copy|cpy)\s+async\b", "cudaMemcpyAsync", "cuda"),
    _rule("cuda_memcpy", r"\bcuda\s*mem\s*(?:copy|cpy)\b", "cudaMemcpy", "cuda"),
    _rule("cuda_free", r"\bcuda\s+free\b", "cudaFree", "cuda"),
    _rule("q_d_a_cuda", r"\bq\s*d\s*a\b", "CUDA", "cuda"),
    _rule("number_gb", r"\b(\d+(?:\.\d+)?)\s+gigabytes\b", r"\1 GB", "units"),
    _rule("torch_cuda_is_available", r"\btorch\s*(?:\.|dot|\s+)\s*cuda\s*(?:\.|dot|\s+)\s*is\s*[._ ]?available\b", "torch.cuda.is_available", "api"),
    _rule("at_cuda_jit", r"\bat\s+cuda\s*(?:\.|dot)\s*jit\b|\b@\s*cuda\s*(?:\.|dot)\s*jit\b", "@cuda.jit", "api"),
    _rule("reinterpret_cast", r"\bre[-\s]*interpret\s+cast\b|\breinterpret\s+cast\b", "reinterpret_cast", "cpp"),
    _rule("static_cast", r"\bstatic\s+cast\b", "static_cast", "cpp"),
    _rule("dynamic_cast", r"\bdynamic\s+cast\b", "dynamic_cast", "cpp"),
    _rule("use_fast_math", r"\buse\s+fast\s+math\b", "use_fast_math", "cuda"),
    _rule("app_post_generate", r"\badd\s+app\s+post\s+slash\s+generate\b", '@app.post("/generate")', "code"),
    _rule("generate_text_sig", r"\bgenerate\s+underscore\s+text\s+prompt\s+str\b", "generate_text(prompt: str)", "code"),
    _rule("vast_ai", r"\bsebastia\b(?=\s+is\s+awesome\b)", "vast.ai", "service"),
    _rule("rest_endpoint", r"\brest\s+endpoint\b", "REST endpoint", "api"),
    _rule("langchain", r"\blang\s*chain\b", "LangChain", "framework"),
    _rule("scikit_learn", r"\bscikit\s+learn\b|\bsk\s+learn\b", "Scikit-learn", "framework"),
    _rule("sklearn", r"\bsk\s*learn\b", "sklearn", "package"),
)


def _separator_symbol(separator: str) -> str:
    if "dash" in separator.lower() or "hyphen" in separator.lower():
        return "-"
    if "underscore" in separator.lower() and len(separator.lower().split()) >= 2:
        return "__"
    return "_"


def _token_is_identifierish(token: str) -> bool:
    if re.search(r"\d|\.", token):
        return True
    if re.search(r"[a-z][A-Z]", token):
        return True
    if re.fullmatch(r"[A-Z]{2,}", token):
        return True
    return bool(re.fullmatch(r"[A-Z]", token))


def _collapse_spoken_separator_run(match: re.Match[str]) -> str:
    raw = match.group(0)
    parts = SPOKEN_SEPARATOR_SPLIT_RE.split(raw)
    tokens = parts[::2]
    separators = parts[1::2]
    if len(tokens) < 2 or len(tokens) != len(separators) + 1:
        return raw

    repeated_pattern = len(separators) >= 2
    has_code_shape = any(_token_is_identifierish(token) for token in tokens)
    symbols = [_separator_symbol(separator) for separator in separators]

    if any(symbol == "-" for symbol in symbols):
        for index, symbol in enumerate(symbols):
            if symbol == "-" and not (
                _token_is_identifierish(tokens[index])
                and _token_is_identifierish(tokens[index + 1])
            ):
                return raw

    if not (has_code_shape or repeated_pattern):
        return raw

    collapsed = tokens[0]
    for symbol, token in zip(symbols, tokens[1:], strict=True):
        collapsed += symbol + token
    return collapsed


def _collapse_spoken_separators(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        replacement = _collapse_spoken_separator_run(match)
        changed = changed or replacement != match.group(0)
        return replacement

    return SPOKEN_SEPARATOR_RUN_RE.sub(replace, text), changed


def _small_integer_value(token: str) -> int | None:
    if token.isdigit():
        value = int(token)
    else:
        value = SMALL_INTEGER_WORDS.get(token.lower())
    if value is None or value > 12:
        return None
    return value


def _collapse_dimension_by(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        left = _small_integer_value(match.group(1))
        right = _small_integer_value(match.group(2))
        if left is None or right is None:
            return match.group(0)
        context = text[max(0, match.start() - 48) : min(len(text), match.end() + 48)]
        if not DIMENSION_CONTEXT_RE.search(context):
            return match.group(0)
        changed = True
        return f"{left}x{right}"

    return DIMENSION_BY_RE.sub(replace, text), changed


def _cleanup_code_spacing(text: str) -> str:
    text = re.sub(
        r"\b(AutoTokenizer|tokenizer|model)\s*(?:\.|dot)\s*(from_pretrained|save_pretrained)\b",
        r"\1.\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(AutoTokenizer|tokenizer|model)\s*(?:\.|dot)\s*from\s+pre[- ]trained\b",
        r"\1.from_pretrained",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(AutoTokenizer|tokenizer|model)\s*(?:\.|dot)\s*save\s+pre[- ]trained\b",
        r"\1.save_pretrained",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bautotokenizer\.from_pretrained\b", "AutoTokenizer.from_pretrained", text, flags=re.IGNORECASE)
    text = re.sub(r"\btokenizer\.save_pretrained\b", "tokenizer.save_pretrained", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmodel\.save_pretrained\b", "model.save_pretrained", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmodel\.from_pretrained\b", "model.from_pretrained", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrom\s+underscore\s+pt\b", "from_pt", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpip\s+install\s+-\s+r\b", "pip install -r", text, flags=re.IGNORECASE)
    text = re.sub(r"\bapplication\s*/\s*slash\s+json\b", "application/json", text, flags=re.IGNORECASE)
    text = re.sub(r"\bapplication\s+slash\s+json\b", "application/json", text, flags=re.IGNORECASE)
    return text


def apply_exact_form_corrections(text: str) -> tuple[str, list[str]]:
    corrected = str(text or "")
    applied: list[str] = []
    for rule in RULES:
        next_text, count = rule.pattern.subn(rule.replacement, corrected)
        if count:
            applied.append(rule.name)
            corrected = next_text
    corrected, count = _collapse_spoken_separators(corrected)
    if count:
        applied.append("spoken_identifier_separators")
    corrected, count = _collapse_dimension_by(corrected)
    if count:
        applied.append("spoken_dimension_by")
    corrected = _cleanup_code_spacing(corrected)
    corrected = re.sub(r"[ \t]+", " ", corrected).strip()
    return corrected, applied


def correction_rule_names() -> list[str]:
    return [rule.name for rule in RULES]
