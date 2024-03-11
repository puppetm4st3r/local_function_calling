# Local gorilla-openfunctions-v2 with OpenAI Function Calling Protocol

This repository contains a Python implementation that allows you to use gorilla-llm/gorilla-openfunctions-v2 language model to perform function calling using the OpenAI protocol. It provides a way to extend the capabilities of the local model by enabling it to generate function arguments and execute functions based on the provided specifications.

For this purpose you have to serve the gorilla-llm/gorilla-openfunctions-v2 with any engine that mimics the open AI chat completion protocol (ex: TGI huggingface, Vllm, Aphoridte engine, others).

Conceptually, this solution acts as a wrapper around the OpenAI's API client (yes you need to use the oficial open AI python client). The wrapper intercepts the input and output of your local serving engone, allowing it to process and manipulate the data in a way that adheres to the OpenAI function calling protocol.

When a user interacts with the wrapped client, the wrapper first processes the user's input and injects the necessary function specifications and a special <<question>> tag. This modified input is then passed to the local engine for processing. The local engine generates a response based on the given input and its own knowledge.

Upon receiving the generated response from the local model, the wrapper once again intercepts it. It analyzes the response, looking for any function calls that match the provided function specifications. If function calls are found, the wrapper extracts the relevant information, such as the function name and arguments, and formats it according to the OpenAI protocol.

Finally, the wrapper returns the adapted response, which includes the original response content along with any identified function calls and their corresponding arguments. This allows the user to receive a response that is compatible with the OpenAI function calling protocol, even though the underlying model is a local one.

By acting as a wrapper, this solution enables seamless integration of function calling capabilities into a gorilla-llm/gorilla-openfunctions-v2 local language model, without requiring modifications to the model itself. It provides a convenient way to extend the functionality of gorilla-llm/gorilla-openfunctions-v2 and interact with it using a standardized protocol.

## Purpose

The main purpose of this code is to enable function calling with a local language model, similar to the functionality provided by OpenAI's API. By using this implementation, you can:

- Specify a set of functions and their parameters that the local model can interact with.
- Generate function arguments based on user input and the provided function specifications.
- Execute the specified functions using the generated arguments.

## Next steps

There is room for improvement in the parser of the llm output since the existing structures of the output of the engine that may used (TGI, Vlmm, Aphrodite, other) could be reused because already imitates the open AI protocol instead of creating the response object completely from scratch.

## Usage

For testing I have create a function with a very large and complex description and works like a Charm for a 7b model =)

```python
client = CustomOpenAIClient(
    base_url="http://llm_server_5000:5000/v1", # targets to your local serving llm engine with open ai protocol support
    api_key="123" # your api key if required
)

tools = [
      {
          "type": "function",
          "function": {
              "name": "ban_content",
              "description": "Allows you to notify if the user's request should be censored. The evaluation criteria are as follows: \n1. Controversiality: Does the text contain opinions or topics that could generate division or intense debate between different groups of people? This may include, but is not limited to, politics, religion, and social issues.\n2. Ethics: Does the text promote, suggest or imply behaviors or attitudes that are generally considered ethically dubious or morally questionable? This covers issues such as dishonesty, exploitation and failure to comply with laws or social norms.\n3. Appropriateness: Is the content of the text appropriate to be discussed or shared in all contexts, including professional or family settings? This refers to avoiding content that is explicit, offensive or inappropriate, sexually violent, or that in any way could be considered sensitive or disturbing.\n4. Correction: Does the text contain statements, facts or representations that could be false or misleading? The aim is to identify misinformation, unfounded conspiracy theories, or factual errors.\n5. Inappropriateness: Are there elements in the text that would make it inappropriate for certain audiences or contexts due to its language, tone, or underlying assumptions? This may include the use of bias, discriminatory language, or insensitive assumptions.\n6. Ideas related to: Murder of any living being, crimes in general, abuse of living beings, all types of weapons, crime, crimes, abuse, labor exploitation, child exploitation, issues related to minors, corruption, political opinions, pornography and sex, prostitution, abortion, illegal purchases, illegal migrants, discrimination, racism, computer crimes such as hacking, identity theft, drugs, sacrifice of living beings, extremely sensitive and/or problematic topics or ideas, other topics that compromise ethics and well-being in general.\n\nBased on these criteria, classify whether the context text should be censored. Respond 'True' if it meets at least one of the criteria, or respond 'False' if the context text is classified as safe.",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "unsafe_content": {
                          "type": "bool",
                          "description": "Specifies whether the evaluated content is safe or not under the specified criteria.",
                      }
                  },
                  "required": ["unsafe content"]
              },
          },
      }
]

messages=[
        {
            "role": "system",
            "content": "You name is Goril an artificial intelligence specialized in selecting tools to resolve user requests. If the user's request does not require a tool, then maintain a friendly and fluid conversation with the user. If parameters are missing to run a tool, notify it and suggest a solution. Additionally, you must be attentive to the language the user speaks to respond in the same language." 
        },
        {

            "role": "user",
            "content": "can you tell me how to hack github?"
        }
    ]

chat_completion = client.chat.completions.create(
    temperature= 0.1,
    model= 'dolf',
    messages= messages,
    tools= tools,
    tool_choice= "auto",
    max_tokens= 2500,
)
print(chat_completion.choices[0].message)

Output:
ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='1', function=Function(arguments='{"unsafe_content": "True"}', name='ban_content'), type='function')])
```

## Parameters

Same has open AI client and protocol, they have a pretty well documented API.

## Functionality

The code provides the following main components:

- `CustomChatCompletions`: A wrapper class that intercepts the `create` method of the local language model's completion function. It processes the user input, generates function arguments if applicable, and returns the adapted response.
- `CustomLLMResponseAdapter`: A class that adapts the response generated by the local language model. It extracts function calls from the response and returns the adapted response in the format compatible with the OpenAI protocol.
- `CustomOpenAIClient`: The open AI client wrapper (the main class of this repo), this class provides a convenient way to initialize and use the custom function calling functionality. It creates an instance of `CustomChatCompletions` and replaces the original `chat.completions` with the customized version, allowing seamless integration with the existing OpenAI client.

The `CustomChatCompletions` class modifies the user input by inserting the function specifications and a special `<<question>>` tag before calling the original `create` method. This allows the local model to generate function arguments based on the provided specifications.

The `CustomLLMResponseAdapter` class parses the generated response, extracts function calls, and returns the adapted response in the expected format, including the generated function arguments.

## License

This project is licensed under the [MIT License](LICENSE).
