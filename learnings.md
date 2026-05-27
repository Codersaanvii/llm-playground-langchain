#using langchain to create a llm chat model
learnt models, PromptTemplates,Chains,Memory,Output Parsers, Modes

creates reusable prompts using PromptTemplate-> this allows placeholders for {history},{input},{system_prompt}

LLMChain connects model, memory and prompt into one chain.predict()- which inserts var in template, retrieves memory, builds prompt on the basis on a mode selected, sends request to model and returns a reponse

experimented with the various conversation memory types-

1. ConversationBufferMemory - stores entire conversations
2. ConversationBufferWindowMemory - stores recent interactions based on k value (sliding window memory) it is a short term memory
3. ConversationTokenBufferMemory - uses token length rather than number of interactions
4. ConversationSummaryBufferMemory - summarizes old conversations and stores them , stores important info only, summary memory relies on token counting.

created multiple assistant personalities by experimenting with prompt modes- this controlled the behavior and output of the model

structured output parsing- used schemas to create response blueprints
parsers generate formatting instructions
LLM outputs can be converted to python dict
