from typing import List

import fitz
import openai
import pandas as pd
from tqdm import tqdm


class PDFQnAGenerator:
    def __init__(self, pdf_path: str, openai_api_key: str):
        self.pdf_path = pdf_path
        self.openai_api_key = openai_api_key
        self.scraped_content = self.read_pdf_content()
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.raw_content_questions = []
        self.raw_content_answers = []

    def read_pdf_content(self) -> List[str]:
        """
        Reads a PDF and returns its content as a list of strings.

        Returns:
        list of str: A list where each element is the text content of a PDF page.
        """
        content_list = []
        with fitz.open(self.pdf_path) as doc:
            for page in doc:
                content_list.append(page.get_text())

        return content_list

    def process_scraped_content(self):
        """
        Process scraped content to replace special characters and split into sentences.
        """
        self.scraped_content = " ".join(self.scraped_content)
        self.scraped_content = [
            self.scraped_content.split(". ")[i]
            .replace("\n", "")
            .replace("   ", "")
            .replace("  ", "")
            for i in range(len(self.scraped_content.split(". ")))
        ]

    def call_chatgpt(self, query: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Generates a response to a query using the specified language model.
        Args:
            query (str): The user's query that needs to be processed.
            model (str, optional): The language model to be used. Defaults to "gpt-3.5-turbo".
        Returns:
            str: The generated response to the query.
        """

        # Prepare the conversation context with system and user messages.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {query}."},
        ]

        # Use the OpenAI client to generate a response based on the model and the conversation context.
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )

        # Extract the content of the response from the first choice.
        content: str = response.choices[0].message.content

        # Return the generated content.
        return content

    def prompt_engineered_api(self, text: str) -> str:
        """
        Generate a question based on the provided text content.
        """
        prompt = f"""
            I have the following content: {text}

            Write one question based on the content above. Just write ONE question in a sentence. No more.
        """

        resp = self.call_chatgpt(prompt)

        return resp

    def generate_questions_answers(self):
        """
        Generate questions and answers from the scraped content.
        """
        for i in tqdm(range(len(self.scraped_content))):
            quest = self.scraped_content[i]
            resp = self.prompt_engineered_api(quest)
            this_sample_question = resp.split("###")[0]
            this_sample_answer = self.scraped_content[i]
            self.raw_content_questions.append(this_sample_question)
            self.raw_content_answers.append(this_sample_answer)

    def convert_to_dataframe(self) -> pd.DataFrame:
        """
        Converts a list of questions and answers into a Pandas DataFrame.

        Returns:
            - Pandas DataFrame: The resulting data frame with columns for each question-answer pair.
        """

        # Convert lists to Series objects for easier indexing
        qns_series = pd.Series(
            [question + "\n" for question in self.raw_content_questions]
        )
        ans_series = pd.Series(self.raw_content_answers)

        # Create a data frame from the Series objects
        df = pd.DataFrame({"questions": qns_series, "answers": ans_series})

        # Reshape the data frame so that it has one row for each question and its corresponding answer. Drop any rows where there are no answers provided.
        df = df.explode("questions").reset_index().dropna()

        # Save a .csv file
        file_path_collapsed = self.pdf_path.replace("/", "_").replace(" ", "_")
        df.to_csv(f"questions_answers__{file_path_collapsed}.csv", index=False)

        return df
