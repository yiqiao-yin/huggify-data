import os
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from openai import OpenAI  # Ensure you have the correct OpenAI client installed


class ChatBot:
    def __init__(self, api_key: str):
        """
        Initializes the ChatBot with an OpenAI client and conversation history.
        """
        self.client = OpenAI(
            api_key=api_key
        )  # Initialize the OpenAI client with API key
        self.history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]  # Initialize conversation history

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the chatbot based on the user's prompt.

        Args:
        - prompt (str): The user's input prompt.

        Returns:
        - str: The chatbot's response.
        """
        # Append the user prompt to the conversation history
        self.history.append({"role": "user", "content": prompt})

        # Generate a response using the OpenAI API
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Model used for generating the response
            messages=self.history,  # Pass the conversation history
        )

        # Extract the response from the API output
        response = completion.choices[0].message.content
        # Append the response to the conversation history
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_history(self) -> List[dict]:
        """
        Retrieves the conversation history.

        Returns:
        - List[dict]: The conversation history as a list of message dictionaries.
        """
        return self.history

    def convert_to_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """
        Converts a list of sentences into a list of numerical embeddings using OpenAI's embedding model.

        Args:
        - sentences (List[str]): A list of sentences (strings).

        Returns:
        - List[List[float]]: A list of lists of numerical embeddings.
        """
        embeddings = []  # Initialize the list to store embeddings

        # Loop through each sentence to convert to embeddings
        for sentence in sentences:
            # Use the OpenAI API to get embeddings for the sentence
            response = self.client.embeddings.create(
                input=sentence,
                model="text-embedding-ada-002",  # Ensure the correct model is used
            )

            # Append the embedding to the list
            embeddings.append(response.data[0].embedding)

        return embeddings

    @staticmethod
    def quantize_to_kbit(arr: Union[np.ndarray, Any], k: int = 16) -> np.ndarray:
        """Converts an array to a k-bit representation by normalizing and scaling its values.
        Args:
            arr (Union[np.ndarray, Any]): The input array to be quantized.
            k (int): The number of levels to quantize to. Defaults to 16 for 4-bit quantization.
        Returns:
            np.ndarray: The quantized array with values scaled to 0 to k-1.
        """
        if not isinstance(arr, np.ndarray):  # Check if input is not a numpy array
            arr = np.array(arr)  # Convert input to a numpy array
        arr_min = arr.min()  # Calculate the minimum value in the array
        arr_max = arr.max()  # Calculate the maximum value in the array
        normalized_arr = (arr - arr_min) / (
            arr_max - arr_min
        )  # Normalize array values to [0, 1]
        return np.round(normalized_arr * (k - 1)).astype(
            int
        )  # Scale normalized values to 0-(k-1) and convert to integer

    @staticmethod
    def quantized_influence(
        arr1: np.ndarray, arr2: np.ndarray, k: int = 16, use_dagger: bool = False
    ) -> Tuple[float, List[float]]:
        """
        Calculates a weighted measure of influence based on quantized version of input arrays and optionally applies a transformation.
        Args:
            arr1 (np.ndarray): First input array to be quantized and analyzed.
            arr2 (np.ndarray): Second input array to be quantized and used for influence measurement.
            k (int): The quantization level, defaults to 16 for 4-bit quantization.
            use_dagger (bool): Flag to apply a transformation based on local averages, defaults to False.
        Returns:
            Tuple[float, List[float]]: A tuple containing the quantized influence measure and an optional list of transformed values based on local estimates.
        """
        # Quantize both arrays to k levels
        arr1_quantized = ChatBot.quantize_to_kbit(arr1, k)
        arr2_quantized = ChatBot.quantize_to_kbit(arr2, k)

        # Find unique quantized values in arr1
        unique_values = np.unique(arr1_quantized)

        # Compute the global average of quantized arr2
        total_samples = len(arr2_quantized)
        y_bar_global = np.mean(arr2_quantized)

        # Compute weighted local averages and normalize
        weighted_local_averages = [
            (np.mean(arr2_quantized[arr1_quantized == val]) - y_bar_global) ** 2
            * len(arr2_quantized[arr1_quantized == val]) ** 2
            for val in unique_values
        ]
        qim = np.sum(weighted_local_averages) / (
            total_samples * np.std(arr2_quantized)
        )  # Calculate the quantized influence measure

        if use_dagger:
            # If use_dagger is True, compute local estimates and map them to unique quantized values
            local_estimates = [
                np.mean(arr2_quantized[arr1_quantized == val]) for val in unique_values
            ]
            daggers = {
                unique_values[i]: v for i, v in enumerate(local_estimates)
            }  # Map unique values to local estimates

            def find_val_(i: int) -> float:
                """Helper function to map quantized values to their local estimates."""
                return daggers[i]

            # Apply transformation based on local estimates
            daggered_values = list(map(find_val_, arr1_quantized))
            return qim, daggered_values
        else:
            # If use_dagger is False, return the original quantized arr1 values
            daggered_values = arr1_quantized.tolist()
            return qim, daggered_values

    def rag(self, prompt: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Searches a predefined database by converting the prompt and database entries to embeddings,
        and then calculating a quantized influence metric.

        Args:
        - prompt (str): A text prompt to search for in the database.
        - df (pd.DataFrame): A pandas DataFrame containing a 'questions' column.

        Returns:
        - pd.DataFrame: A pandas DataFrame sorted by the quantized influence metric in descending order.
                        The DataFrame contains the original questions, their embeddings, and the computed scores.
        """
        # Convert the prompt to its numerical embedding using the bot's embedding function
        prompt_embedding = self.convert_to_embeddings([prompt])[0]

        # Convert the 'questions' column to numerical embeddings and store in a new column 'questions_embed'
        df["questions_embed"] = self.convert_to_embeddings(df["questions"].tolist())

        # Compute similarity scores using quantized_influence and store in a new column 'similarity_score'
        df["similarity_score"] = [
            self.quantized_influence(
                prompt_embedding, df["questions_embed"][i], k=16, use_dagger=False
            )[0]
            for i in range(len(df))
        ]

        # Sort the DataFrame based on the 'similarity_score' in descending order
        sorted_df = df.sort_values(by="similarity_score", ascending=False)

        return sorted_df

    def run_rag(
        self, openai_api_key: str, current_prompt: str, df: pd.DataFrame, top_n: int = 2
    ) -> dict:
        """
        Executes the RAG process and generates a response based on the top N results.

        Args:
        - openai_api_key (str): The OpenAI API key.
        - current_prompt (str): The current prompt to query.
        - df (pd.DataFrame): The DataFrame containing the 'questions' and 'answers' columns.
        - top_n (int): The number of top results to consider for generating the final response.

        Returns:
        - dict: A dictionary containing the generated response and the top N references as a DataFrame.
                {"response": str, "references": pd.DataFrame}
        """
        # Initialize the ChatBot
        bot = ChatBot(api_key=openai_api_key)

        # Perform RAG to get the sorted DataFrame
        df_sorted = bot.rag(current_prompt, df)

        # Select the top N results
        df_short = df_sorted[["questions", "answers"]].head(top_n)

        # Construct the references from the top N results
        references = " ".join(
            [
                "Q: "
                + df_short["questions"].to_list()[i]
                + "; A: "
                + df_short["answers"].to_list()[i]
                for i in range(top_n)
            ]
        )

        # Generate the final response based on the references
        response = bot.generate_response(
            f"""
            Answer the question: {current_prompt} based on the following reference:

            {references}
        """
        )

        # Return the response and df_short in a dictionary
        return {"response": response, "references": df_short}
