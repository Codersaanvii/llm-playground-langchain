import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
load_dotenv()

#first you create a model
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)
#create modes for assistant
modes ={
    "default": "you are a kind and helpful AI assistant who explains things in a concise manner",
    "strict": "you are a strict AI assistant who is stern and mean and gives direct answers",
    "coder":"you are a senior and experienced AI and ML engineer with a softare engineering background who explains code clearly",
    "motivator":"you are a motivational coach who guides and encourages the user",
    "json":"you are a json formatter and a data extraction assistant"
}

#.get() is a method of dict that returns key value if key is present
mode = input("choose mode (default/strict/coder/motivator/json): ")

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

{format_instructions}
AI:
"""
#create a prompt template object this is expected by LLMChain
prompt = PromptTemplate(
    input_variables = ["history", "input","system_prompt","format_instructions"],
    template = template
)

#memory created
memory = ConversationSummaryBufferMemory(
    llm = llm,
    memory_key = "history",
    input_key = "input",
    max_token_limit = 20
)

#output parser
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

#creating schema -> this defines the structure of output from AI (response blueprint)
topic_schema = ResponseSchema(
    name ="topic",
    description = "main topic discussed by user"
)
sentiment_schema = ResponseSchema(
    name = "sentiment",
    description = "emotion or sentiment of user message : postive, negative or neutral"
)
summary_schema = ResponseSchema(
    name="summary",
    description="concise summary of user message"
)
response_schemas = [
    topic_schema,
    sentiment_schema,
    summary_schema
]

#creates a parser object by taking all the schemas - from_response_schemas() is a langchain class(StructuredOutputParser is a class) method (method attached to class)
output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)

#generates format instructions for AI-> prompt instructions
if mode == "json":
    format_instructions = output_parser.get_format_instructions()
else:
    format_instructions = ""

#create a chain which holds together model, template and memory
from langchain.chains import LLMChain
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

    response = chain.predict(input = user_input, system_prompt = system_prompt,    format_instructions = format_instructions)
    print("AI:", response)

    if mode == "json":
        #actual parsing happens -> parse() converts the str to python data
        parsed_output = output_parser.parse(response)
        print(parsed_output)


print(memory.buffer)