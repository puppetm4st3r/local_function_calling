from openai import OpenAI
from typing import Dict, Any, Union
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import json

class CustomLLMResponseAdapter:
    @classmethod
    def adapt_response(cls, response: str, completion_kwargs: Dict[str, Any] = {}) -> Union[ChatCompletion, Choice]:
        completion_kwargs = completion_kwargs or {}
        function_calls: List[ChatCompletionMessageToolCall] = []
        if "<<function>>" in response:
            function_parts = response.split("<<function>>")
            for part in function_parts[1:]:
                if "(" in part:
                    function_name, function_args_str = part.split("(", 1)
                    function_args_str = function_args_str.rstrip(")")
                    function_args = {}
                    for arg in function_args_str.split(","):
                        arg_name, arg_value = arg.split("=")
                        function_args[arg_name.strip()] = arg_value.strip()
                    function_calls.append(ChatCompletionMessageToolCall(
                        id=completion_kwargs.get("tool_call_id", "1"),
                        type="function",
                        function=Function(
                            name=function_name.strip(),
                            arguments=json.dumps(function_args)
                        )
                    ))

        usage = CompletionUsage(
            prompt_tokens=completion_kwargs.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=completion_kwargs.get("usage", {}).get("completion_tokens", 0),
            total_tokens=completion_kwargs.get("usage", {}).get("total_tokens", 0)
        )

        if function_calls:
            return ChatCompletion(
                id=completion_kwargs.get("id", "chatcmpl-default-id"),
                object="chat.completion",
                created=completion_kwargs.get("created", 0),
                model=completion_kwargs.get("model", "default-model"),
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            function_call=None,
                            tool_calls=function_calls
                        )
                    )
                ],
                usage=usage
            )
        else:
            return ChatCompletion(
                id=completion_kwargs.get("id", "chatcmpl-default-id"),
                object="chat.completion",
                created=completion_kwargs.get("created", 0),
                model=completion_kwargs.get("model", "default-model"),
                choices=[
                    Choice(
                        finish_reason=completion_kwargs.get("finish_reason", "stop"),
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=response,
                            function_call=None,
                            tool_calls=None
                        )
                    )
                ],
                usage=usage
            )

class CustomChatCompletions:
    def __init__(self, completions):
        self._original_completions = completions

    def create(self, *args, **kwargs):
        messages = kwargs.get("messages", None)
        if messages is None:
            for arg in args:
                if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], dict) and "role" in arg[0]:
                    messages = arg
                    break

        tools = kwargs.get("tools", None)
        if tools is None:
            for arg in args:
                if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], dict) and "type" in arg[0]:
                    tools = arg
                    break

        if messages is not None and tools is not None:
            # Obtener la cadena de funciones de 'tools'
            functions_string = json.dumps(tools)
            
            updated_messages = self.insert_function_and_question(messages, functions_string)
            args = tuple(updated_messages if arg is messages else arg for arg in args)
            kwargs["messages"] = updated_messages

        response = self._original_completions.create(*args, **kwargs)

        adapted_response = CustomLLMResponseAdapter.adapt_response(response.choices[0].message.content)
        return adapted_response

    def __getattr__(self, item):
        return getattr(self._original_completions, item)

    @staticmethod
    def insert_function_and_question(messages, functions_string):
        user_message = None
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message
                break

        if user_message:
            user_message["content"] = f"<<function>>{functions_string}\n<<question>>{user_message['content']}"

        return messages

class CustomOpenAIClient(OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat.completions = CustomChatCompletions(self.chat.completions)
