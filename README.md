# huggify-data

## Introduction

**huggify-data** 📦 is a Python library 🐍 designed to simplify the process of scraping `.pdf` documents, generating question-answer pairs using `openai`, converse with the document, and then uploading datasets 📊 to the Hugging Face Hub 🤗. This library allows you to verify ✅, process 🔄, and push 🚀 your pandas DataFrame directly to Hugging Face, making it easier to share and collaborate 🤝 on datasets. Additionally, the new version enables users to fine-tune the Llama2 model on their proprietary data, enhancing its capabilities even further. As the name suggests, the **huggify-data** package enhances your data experience by wrapping it in warmth, comfort, and user-friendly interactions, making data handling feel as reassuring and pleasant as a hug.

[![Watch the video](https://img.youtube.com/vi/XLExhyangWw/0.jpg)](https://www.youtube.com/watch?v=XLExhyangWw)

## Installation

To use **huggify-data**, ensure you have the necessary libraries installed. You can easily install them using pip:

```sh
pip install huggify-data
```

## Notebooks

We have made tutorial notebooks available to guide you through the process step-by-step:

- **Step 1**: Scrape any `.pdf` file and generate question-answer pairs. [Link](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex_%20-%20huggify%20data%20-%20part%201%20-%20scrape%20and%20generate%20qa.ipynb)
- **Step 2**: Fine-tune the Llama2 model on customized data. [Link](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex_%20-%20huggify%20data%20-%20part%202%20-%20fine%20tune%20llama2%20over%20custom%20data.ipynb)
- **Step 3**: Perform inference on customized data. [Link](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex_%20-%20huggify%20data%20-%20part%203%20-%20inference%20using%20fine%20tuned%20llama2.ipynb)

## Examples

Here's a complete example illustrating how to use **huggify-data** to scrape a PDF and save it as question-answer pairs in a `.csv` file. The following block of code will scrape the content, convert it into a `.csv`, and save the file locally:

```python
from huggify_data.scrape_modules import *

# Example usage:
pdf_path = "path_of_pdf.pdf"
openai_api_key = "<sk-API_KEY_HERE>"
generator = PDFQnAGenerator(pdf_path, openai_api_key)
generator.process_scraped_content()
generator.generate_questions_answers()
df = generator.convert_to_dataframe()
print(df)
```

Once you have created a data frame of question-answer pairs, you can have a conversation with your data:

```python
from huggify_data.bot_modules import *

current_prompt = "<question_about_the_document>"
chatbot = ChatBot(api_key=openai_api_key)
response = chatbot.run_rag(openai_api_key, current_prompt, df, top_n=2)
print(response)
```

Moreover, you can push it to the cloud. Here's a complete example illustrating how to use the **huggify-data** library to push data (assuming an existing `.csv` file with columns `questions` and `answers`) to Hugging Face Hub:

```python
from huggify_data.push_modules import DataFrameUploader

# Example usage:
df = pd.read_csv('/content/toy_data.csv')
uploader = DataFrameUploader(df, hf_token="<huggingface-token-here>", repo_name='<desired-repo-name>', username='<your-username>')
uploader.process_data()
uploader.push_to_hub()
```

Here's a complete example illustrating how to use the **huggify-data** library to fine-tune a Llama2 model (assuming you have a directory from Hugging Face ready):

```python
from huggify_data.train_modules import *

# Parameters
model_name = "NousResearch/Llama-2-7b-chat-hf" # Recommended base model
dataset_name = "eagle0504/sample_toy_data_v9" # Desired name, e.g., <hf_user_id>/<desired_name>
new_model = "youthless-homeless-shelter-web-scrape-dataset-v4" # Desired name
huggingface_token = userdata.get('HF_TOKEN')

# Initiate
trainer = LlamaTrainer(model_name, dataset_name, new_model, huggingface_token)
peft_config = trainer.configure_lora()
training_args = trainer.configure_training_arguments(num_train_epochs=2)

# Training
trainer.train_model(training_args, peft_config)

# Train and save | Run this in a new cell
trainer.merge_and_save_model()
```

To perform inference, please follow the example below:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="eagle0504/youthless-homeless-shelter-web-scrape-dataset-v4") # Same name as above
response = pipe("### Human: What is YSA? ### Assistant: ")
print(response[0]["generated_text"])
print(response[0]["generated_text"].split("### ")[-1])
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yiqiao-yin/huggify-data/blob/main/LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

## Contact

For any questions or support, please contact [eagle0504@gmail.com](mailto: eagle0504@gmail.com).