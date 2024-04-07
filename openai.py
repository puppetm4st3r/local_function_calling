from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, List, cast
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.resources.chat.completions import Completions, AsyncCompletions
from openai._streaming import  Stream, AsyncStream

import json
import re

class CustomLLMResponseAdapter:

    arg_pattern = re.compile(r'(\w+)=((?:\'(?:[^\']|\'\')*?\'|"(?:[^"]|"")*?"|\S+?)(?:,|$))')

    @classmethod
    def adapt_response(cls, response: str, completion_kwargs: Dict[str, Any] = {}) -> ChatCompletion:
        
        def parse_function_args(function_args_str: str) -> Dict[str, Any]:
            function_args = {}
            matches = cls.arg_pattern.findall(function_args_str)
            for match in matches:
                arg_name, arg_value = match
                arg_value = arg_value.strip("'\",")
                arg_value = arg_value.replace("''", "'").replace('""', '"')
                try:
                    parsed_value = json.loads(arg_value)
                    function_args[arg_name] = parsed_value
                except json.JSONDecodeError:
                    function_args[arg_name] = arg_value
            return function_args
        
        completion_kwargs = completion_kwargs or {}
        function_calls: List[ChatCompletionMessageToolCall] = []

        if "<<function>>" in response:
            function_parts = response.split("<<function>>")
            for part in function_parts[1:]:
                if "(" in part:
                    function_name, function_args_str = part.split("(", 1)
                    function_args_str = function_args_str.rstrip(")")
                    function_args = parse_function_args(function_args_str)
                    function_calls.append(ChatCompletionMessageToolCall(
                        id=completion_kwargs.get("tool_call_id", "1"),
                        type="function",
                        function=Function(
                            name=function_name.strip(),
                            arguments= json.dumps(function_args)
                        )
                    ))

        usage = CompletionUsage(
            prompt_tokens=completion_kwargs.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=completion_kwargs.get("usage", {}).get("completion_tokens", 0),
            total_tokens=completion_kwargs.get("usage", {}).get("total_tokens", 0)
        )

        if len(function_calls) > 0:
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
                            content="",
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
                            tool_calls=function_calls
                        )
                    )
                ],
                usage=usage
            )

class CustomChatCompletions:
    def __init__(self, completions:Completions, debug:bool):
        self._original_completions:Completions = completions
        self._debug = debug

    def create(self, *args, **kwargs) -> ChatCompletion | Stream[ChatCompletionChunk]:
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
        
        # check for stream or not
        stream = kwargs.get('stream', True)
        if stream and tools:
            raise(Exception("Stream and function calling is not yet supported."))

        if not stream and tools:
            print('warning: we do not collect token generation metrics here CustomChatCompletions')
            # TODO we do not collect token generation metrics here
            if messages is not None and tools is not None:
                functions_string = json.dumps(tools)
                
                updated_messages = self.insert_function_and_question(messages, functions_string)
                args = tuple(updated_messages if arg is messages else arg for arg in args)
                kwargs["messages"] = updated_messages
            if self._debug: print(f'sending to llm: {updated_messages}')
            response = self._original_completions.create(*args, **kwargs)

            adapted_response = CustomLLMResponseAdapter.adapt_response(cast(str, response.choices[0].message.content)) # return a new ChatCompletion with function callings inside
            if self._debug: print(f'generated by llm: {adapted_response}')
            return adapted_response
        else:
            return self._original_completions.create(*args, **kwargs)

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
        self.chat.completions = cast(Completions, CustomChatCompletions(self.chat.completions, debug= False)) # type: ignore

class AsyncCustomChatCompletions:
    def __init__(self, completions:AsyncCompletions, debug:bool):
        self._original_completions:AsyncCompletions = completions
        self._debug = debug

    async def create(self, *args, **kwargs) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
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
        
        # check for stream or not
        stream = kwargs.get('stream', True)
        if stream and tools:
            raise(Exception("Stream and function calling is not yet supported."))

        if not stream and tools:
            # TODO we do not collect token generation metrics here
            print('warning: we do not collect token generation metrics here AsyncCustomChatCompletions')
            if messages is not None and tools is not None:
                functions_string = json.dumps(tools)
                
                updated_messages = self.insert_function_and_question(messages, functions_string)
                args = tuple(updated_messages if arg is messages else arg for arg in args)
                kwargs["messages"] = updated_messages
            if self._debug: print(f'sending to llm: {updated_messages}')
            response = await self._original_completions.create(*args, **kwargs)

            adapted_response = CustomLLMResponseAdapter.adapt_response(cast(str, response.choices[0].message.content)) # return a new ChatCompletion with function callings inside
            if self._debug: print(f'generated by llm: {adapted_response}')
            return adapted_response
        else:
            return await self._original_completions.create(*args, **kwargs)

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

class AsyncCustomOpenAIClient(AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat.completions = cast(AsyncCompletions, AsyncCustomChatCompletions(self.chat.completions, debug= False)) # type: ignore
