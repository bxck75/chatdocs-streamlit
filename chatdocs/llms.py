from typing import Any, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import CTransformers, HuggingFacePipeline
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .utils import merge


def get_llm(
    config: dict[str, Any],
    *,
    selected_llm_index: int = 0,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> LLM:
    local_files_only = not config["download"]

    selection = config["llms"][selected_llm_index]
    model_framework = selection.pop("model_framework")
    config = {**selection}

    if model_framework == "ctransformers":
        config = merge(config, {"config": {"local_files_only": local_files_only}})
        llm = CTransformers(callbacks=callbacks, **config)
    else:
        config = merge(config, {"model_kwargs": {"local_files_only": local_files_only}})

        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        model = AutoModelForCausalLM.from_pretrained(config["model"], **config["model_kwargs"])
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = model.config.eos_token_id
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            callbacks=callbacks,
            **config["pipeline_kwargs"],
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
    return llm
