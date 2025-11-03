import warnings
from typing import Any, Optional, Union, cast

import torch
from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import (
    FlatMessagesType,
    LLMMessageLogType,
    TaskDataSpec,
)
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_multimodal_keys_from_processor,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.data.llm_message_utils import get_first_index_that_differs, get_images_from_message

Tensor = torch.Tensor
TokenizerType = PreTrainedTokenizerBase


def get_formatted_message_log_llmjp(
    message_log: LLMMessageLogType,
    tokenizer: TokenizerType,
    task_data_spec: TaskDataSpec,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    add_generation_prompt: bool = False,
    tools: Optional[list[dict[str, Any]]] = None,
) -> LLMMessageLogType:
    """Format and tokenize chat messages using the specified template.

    Args:
        message_log: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer for converting text to token IDs
        task_data_spec: Task spec for this dataset.
        add_bos_token: Whether to add bos token to first message if it is not already present. Default: True
        add_eos_token: Whether to add eos token to last message if it is not already present. Default: True
        add_generation_prompt: Whether to include assistant's generation prompt in user messages. Default: False
        tools: Optional list of tool/function definitions to pass to the chat template. Default: None
    Returns:
        The message log with updated 'token_ids' and 'content' fields.
    """
    new_message_log: LLMMessageLogType = []
    prev_formatted_message = ""
    message_log_strs: list[dict[str, str]] = cast(
        list[dict[str, str]], message_log
    )  # we just use the str:str parts here

    multimodal_keys = get_multimodal_keys_from_processor(tokenizer)

    def _format_content_helper(
        content: Union[str, list[dict[str, Any]]],
    ) -> Union[str, list[dict[str, Any]]]:
        """This function formats the text portion of the first user message with the task prompt.

        The `content` argument could either be a string (user text prompt) or a dict (user text prompt + multimodal data).

        Examples of `content` argument include strings or dicts from the following conversation turns:
        - {"role": "user", "content": "What is the capital of France?"}
        - {"role": "user", "content": [{"type": "text", "text": "What is the capital of the city in the image?"}, {"type": "image", "image": "path/to/image.jpg"}]}
        - {"role": "user", "content": [{"type": "text", "text": "Does the animal in the image match the sound it makes in the audio?"}, {"type": "image", "image": "path/to/image.jpg"}, {"type": "audio", "audio": "path/to/audio.mp3"}]}

        In all cases, the text portion of the message is formatted with the task prompt.

        Previously, the `content` argument was modified using
        >>> message_log_strs = [
        ...     {
        ...         "role": "user",
        ...         "content": task_data_spec.prompt.format(message_log_strs[0]["content"]),
        ...     }
        ... ] + message_log_strs[1:]
        >>>

        which assumes that the first message is a string (not true for multimodal data). This helper function correctly handles all cases.
        """
        if isinstance(content, str):
            return task_data_spec.prompt.format(content)
        # this is a list of dicts, format only the text ones
        for item in content:
            if item["type"] == "text":
                item["text"] = task_data_spec.prompt.format(item["text"])
        return content

    # ignore any system prompts
    first_user_msg_id = 0
    for i, msg in enumerate(message_log_strs):
        if msg["role"] == "user":
            first_user_msg_id = i
            break

    if task_data_spec.prompt:
        message_log_strs = (
            message_log_strs[:first_user_msg_id]
            + [
                {
                    "role": "user",
                    "content": _format_content_helper(
                        message_log_strs[first_user_msg_id]["content"]
                    ),
                }
            ]
            + message_log_strs[first_user_msg_id + 1 :]
        )
    
    supported_roles = {"system", "user", "assistant", "tool"}

    for i, message in enumerate(message_log_strs):
        # If enabled, add_generation_prompt is only used on user messages to include
        # the assistant's generation prompt as part of the user message.

        # Only pass tools parameter if tools exist
        template_kwargs = {
            "add_generation_prompt": add_generation_prompt
            and message["role"] in ["user", "tool"],
            "tokenize": False,
            "add_special_tokens": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools

        role = message["role"]
        content = message["content"]
        assert role in supported_roles, (
            f"Unknown message role: {role}. "
            f"Supported roles are: {supported_roles}."
        )

        # formatted_message: str = tokenizer.apply_chat_template(  # type: ignore
        #     message_log_strs[: i + 1], **template_kwargs
        # )

        # ## get the length of the previous message, excluding the eos token (if present)
        # prev_message_len_no_eos: int = get_first_index_that_differs(
        #     prev_formatted_message,
        #     formatted_message,
        # )
        valid_tokenized_offset = 0
        if role == "system":
            message_chunk = content
            valid_tokenized_offset = 0
        elif role == "user":
            message_chunk = f"\n\n### 指示:\n{content}\n\n### 応答:\n"
            valid_tokenized_offset = 1
        elif role == "assistant":
            message_chunk = f"\n{content}"
            valid_tokenized_offset = 2
        elif role == "tool":
            raise NotImplementedError("Tool role is not implemented yet.")

        if i == 0:
            if add_bos_token:
                if tokenizer.bos_token is None:
                    warnings.warn(
                        "add_bos_token is True but the tokenizer does not have a BOS token. Skipping BOS token addition."
                    )
                elif not message_chunk.startswith(tokenizer.bos_token):
                    message_chunk = tokenizer.bos_token + message_chunk

        if i == len(message_log_strs) - 1:
            r"""
            This is an attempt to robustly append the eos token. The origin is Qwen
            chat templates always append <eos>\n and some models like gemma do not
            use the <eos> at all in the chat template. Adding a <eos> if the <eos> is
            already at the end, is likely a user error, and since we know Qwen likes to
            have <eos>\n we'll check for that case.

            This makes the logic slightly more robust to the model family's chat template
            so users don't need to know whether they need to add add_eos or not.
            """
            stripped_message_chunk = message_chunk.rstrip("\n")
            if add_eos_token:
                if tokenizer.eos_token is None:
                    warnings.warn(
                        "add_eos_token is True but the tokenizer does not have an EOS token. Skipping EOS token addition."
                    )
                elif not stripped_message_chunk.endswith(tokenizer.eos_token):
                    message_chunk += tokenizer.eos_token

        # get images too (extend this for other modalities)
        images_cur_message = get_images_from_message(message)

        new_message = message.copy()
        # extend this if statement to check for all(len(modality)) == 0 when adding other modalities
        if len(images_cur_message) == 0:
            new_message["token_ids"] = tokenizer(
                text=message_chunk, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            new_message["token_ids"] = new_message["token_ids"][valid_tokenized_offset:]
        else:
            # extend the else statement to add other modalities (in this case, tokenizer will be a processor)
            processed_chunk = tokenizer(
                text=[message_chunk],
                images=images_cur_message,
                return_tensors="pt",
                add_special_tokens=False,
            )
            new_message["token_ids"] = processed_chunk["input_ids"][0]

            # add all vlm keys to the message
            for key in multimodal_keys:
                if key in processed_chunk:
                    new_message[key] = PackedTensor(processed_chunk[key], dim_to_pack=0)

        if len(new_message["token_ids"]) == 0:
            # if there is an empty message, the empty `token_ids` tensor ends up being in fp32,
            # which causes `_validate_tensor_consistency` to fail. To fix this, we convert the
            # empty tensor to int64.
            new_message["token_ids"] = new_message["token_ids"].to(torch.int64)  # type: ignore

        # format content correctly
        content = message.get("content")
        if content is None or not content:
            # Handle None or missing content (e.g., assistant messages with only tool_calls)
            new_message["content"] = message_chunk
        elif isinstance(content, str):
            new_message["content"] = message_chunk
        else:
            # format the content list of new message the same way as the original message but replace the text with the new message chunk
            new_message["content"] = []
            for item in content:
                if item["type"] == "text":
                    new_message["content"].append(
                        {"type": "text", "text": message_chunk}
                    )
                else:
                    new_message["content"].append(item)

        # Debug: Print each message turn separately (only once for the first sample)
        if not hasattr(get_formatted_message_log_llmjp, "_debug_printed"):
            if i == 0:
                # Print header only at the start of first message
                print("\n" + "=" * 80)
                print("DEBUG: Individual message turns from apply_chat_template")
                print("=" * 80)

            print(f"\n[Turn {i + 1}/{len(message_log_strs)}] Role: {message['role']}")
            print("-" * 40)
            print("Extracted message chunk:")
            print(repr(message_chunk))  # Using repr to show special characters
            print(f"Raw text (len={len(message_chunk)}):")
            print(message_chunk)
            print(f"Valid tokenized offset: {valid_tokenized_offset}")
            print("-" * 40)
            print("Token IDs:")
            print(new_message["token_ids"])
            print("Token strs:")
            print(tokenizer.convert_ids_to_tokens(new_message["token_ids"]))

            if i == len(message_log_strs) - 1:
                # Mark as printed after processing all turns of the first sample
                get_formatted_message_log_llmjp._debug_printed = True

        new_message_log.append(new_message)

    return new_message_log
