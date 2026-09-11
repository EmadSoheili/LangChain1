from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.
    Args:
        query: The query to search for.
    Rteruns:
        The search result.
    """

    print(f"Searching for {query}")
    return "Tokyo wather is sunny"

llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-3.5-flash-lite')
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({'messages':HumanMessage(content = 'What is the wather in Tokyo?')})



if __name__ == "__main__":
    main()