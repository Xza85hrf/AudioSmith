"""Translation module — Argos primary, TranslateGemma optional fallback."""

import logging
from typing import List, Optional

from audiosmith.exceptions import TranslationError
from audiosmith.error_codes import ErrorCode

logger = logging.getLogger(__name__)


def translate_argos(text: str, source_lang: str, target_lang: str) -> str:
    """Translate using argostranslate (offline, fast)."""
    from argostranslate import translate as argos_translate
    return argos_translate.translate(text, source_lang, target_lang)


def translate_gemma(
    text: str,
    source_lang: str,
    target_lang: str,
    model_size: str = 'balanced',
    cache_dir: str = './models/translategemma',
) -> str:
    """Translate using TranslateGemma (GPU, higher quality). Requires [gemma] extra."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_ids = {
        'fast': 'google/translategemma-4b-it',
        'balanced': 'google/translategemma-12b-it',
        'quality': 'google/translategemma-27b-it',
    }
    model_id = model_ids.get(model_size, model_ids['balanced'])

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    try:
        from transformers import BitsAndBytesConfig
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map='auto', cache_dir=cache_dir, quantization_config=qconfig,
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map='auto', cache_dir=cache_dir, torch_dtype=torch.bfloat16,
        )

    messages = [{'role': 'user', 'content': f'Translate from {source_lang} to {target_lang}. Output ONLY the translation.\n\n{text}'}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors='pt', add_generation_prompt=True, return_dict=True,
    ).to(model.device)
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max(input_len * 2, 64), 512),
            do_sample=False,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def translate(text: str, source_lang: str, target_lang: str, backend: str = 'argos') -> str:
    """Translate text. backend='argos' (default, offline) or 'gemma' (GPU).

    Falls back from argos to gemma if argostranslate is not installed.
    """
    if not text or not text.strip():
        return text
    try:
        if backend == 'gemma':
            return translate_gemma(text, source_lang, target_lang)
        return translate_argos(text, source_lang, target_lang)
    except ImportError as e:
        if backend == 'argos':
            logger.warning("argostranslate not available, trying gemma fallback")
            try:
                return translate_gemma(text, source_lang, target_lang)
            except Exception:
                pass
        raise TranslationError(
            f"Translation backend '{backend}' not available: {e}",
            error_code=str(ErrorCode.TRANSLATION_ERROR.value),
            original_error=e,
        )
    except Exception as e:
        raise TranslationError(
            f"Translation failed: {e}",
            error_code=str(ErrorCode.TRANSLATION_ERROR.value),
            original_error=e,
        )


def translate_batch(texts: List[str], source_lang: str, target_lang: str, backend: str = 'argos') -> List[str]:
    """Translate a list of texts sequentially."""
    return [translate(t, source_lang, target_lang, backend) for t in texts]
