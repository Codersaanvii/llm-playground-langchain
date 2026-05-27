import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
load_dotenv()

#first you create a model
llm = ChatOpenAI(
    temperature = 0,
    model = "openai/gpt-3.5-turbo",
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
#create modes for assistant
modes ={
    "default": "you are a kind and helpful AI assistant who explains things in a concise manner",
    "strict": "you are a strict AI assistant who is stern and mean and gives direct answers",
    "french":"you are a french AI assistant who responds politely in french",
    "coder":"you are a senior and experienced AI and ML engineer with a softare engineering background who explains code clearly",
    "motivator":"you are a motivational coach who guides and encourages the user"
}

#.get() is a method of dict that returns key value if key is present
mode = input("choose mode (default/strict/french/coder/motivator): ")

#create system prompt based on mode and inject into template
system_prompt = modes.get(
    mode,
    modes["default"]
)
#then we create a prompt template which has convo history and input of user and system prompt(mode)
template = """
{system_prompt}
Conversation History:
{history}

User:{input}
AI:
"""
#create a prompt template object this is expected by LLMChain
prompt = PromptTemplate(
    input_variables = ["history", "input","system_prompt"],
    template = template
)

#memory created
memory = ConversationBufferMemory(
    memory_key = "history",
    input_key = "input"
)

#create a chain which holds together model, template and memory
from langchain import LLMChain
chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
    verbose = True

)

#loop for conversation 
while True:
    user_input = input("you:")
    if user_input.lower() == "exit":
        break

#chain.predict() takes user input and runs langchain workflow
#Take this input,
#run it through: prompt, memory, model -> then give me the final response text.

    response = chain.predict(input = user_input, system_prompt = system_prompt)
    print("AI:", response)


print(memory.buffer)