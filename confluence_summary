import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from atlassian import Confluence
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up Confluence credentials
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# Initialize Confluence client
confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_TOKEN,
    cloud=True  # Set to False if you're using Confluence Server
)

def get_confluence_page_content(page_id):
    # Fetch the page content
    page = confluence.get_page_by_id(page_id, expand='body.storage')
    
    # Extract the HTML content
    html_content = page['body']['storage']['value']
    
    # Parse HTML and extract text
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text(separator='\n', strip=True)
    
    return text_content

def summarize_text(text, max_length=500):
    # Initialize the OpenAI language model
    llm = OpenAI(temperature=0.7)

    # Define the prompt template
    template = """
    Summarize the following text in a concise manner, highlighting the key points. 
    The summary should be no longer than {max_length} words.

    Text to summarize:
    {text}

    Summary:
    """

    prompt = PromptTemplate(
        input_variables=["text", "max_length"],
        template=template,
    )

    # Create an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    summary = chain.run(text=text, max_length=max_length)

    return summary.strip()

def main():
    # Get the Confluence page ID from user input
    page_id = input("Enter the Confluence page ID: ")

    # Fetch the page content
    content = get_confluence_page_content(page_id)

    # Summarize the content
    summary = summarize_text(content)

    print("\nSummary of the Confluence page:")
    print(summary)

if __name__ == "__main__":
    main()
