import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text): 
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {scrap_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"scrap_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are XYZ, a student currently pursuing your Master's of Science in Data Science at Texas A&M University. With three years of experience in data-related roles, you possess a strong skill set in data analysis, machine learning, and data engineering.
        Your job is to write a cold email to the hiring manager regarding a job opportunity that aligns with your expertise. In your email, express your enthusiasm for the position and highlight your relevant skills and experience that make you a strong candidate.
        Additionally, include examples of past projects or experiences that demonstrate your capabilities in fulfilling the job requirements. 
        Also add the most relevant ones from the following links to showcase XYZ's portfolio: {list_of_links}
        Remember, you are XYZ, a Master's student in Data Science with three years of data experience. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "list_of_links": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))