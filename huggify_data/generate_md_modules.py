import os
import pandas as pd
from typing import List
from openai import OpenAI  # Ensure you have the OpenAI package installed
from .bot_modules import ChatBot  # Import your ChatBot class

class MarkdownGenerator:
    def __init__(self, chatbot: ChatBot, dataframe: pd.DataFrame):
        """
        Initializes the MarkdownGenerator with a ChatBot instance and a DataFrame.

        Args:
        - chatbot (ChatBot): An instance of the ChatBot class.
        - dataframe (pd.DataFrame): The DataFrame containing data to be written to markdown files.
        """
        self.chatbot = chatbot
        self.dataframe = dataframe

    def generate_markdown(self, token_limit: int = 4096, token_threshold: float = 0.7):
        """
        Generates markdown files from the DataFrame, considering token size limits.

        Args:
        - token_limit (int): The token limit for the markdown generation.
        - token_threshold (float): The threshold to determine when to write the markdown file.
        """
        current_token_size = 0
        current_content = ""
        file_count = 1

        for index, row in self.dataframe.iterrows():
            for col in self.dataframe.columns:
                content = f"{col}: {row[col]}\n"
                self.chatbot.generate_response(content)  # Generates response to update history and check token size
                token_size = self._get_token_size(self.chatbot.get_history())
                
                if current_token_size + token_size > token_limit * token_threshold:
                    self._write_to_file(current_content, file_count)
                    file_count += 1
                    current_token_size = token_size
                    current_content = content
                else:
                    current_content += content
                    current_token_size += token_size

        if current_content:
            self._write_to_file(current_content, file_count)

    def _write_to_file(self, content: str, file_count: int):
        """
        Writes the given content to a markdown file.

        Args:
        - content (str): The content to be written to the file.
        - file_count (int): The count of the file being written.
        """
        filename = f"output_{file_count}.md"
        with open(filename, 'w') as file:
            file.write(content)
        print(f"Markdown file {filename} generated.")

    def _get_token_size(self, history: List[dict]) -> int:
        """
        Calculates the token size of the given conversation history.

        Args:
        - history (List[dict]): The conversation history as a list of message dictionaries.

        Returns:
        - int: The total token size of the conversation history.
        """
        return sum(len(message["content"].split()) for message in history)
