
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import pandas as pd

df = pd.read_csv("wec_data.csv")

agent = create_pandas_dataframe_agent(
    ChatGoogleGenerativeAI(model="gemini-pro"),
    df,
    allow_dangerous_code=True,
)

agent.invoke("How many different brands of tyres in the dataset?")
agent.invoke("Show me all different brands of tyres in the dataset")