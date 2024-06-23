# huggify-data

## Introduction

**huggify-data** 📦 is a Python library 🐍 designed to simplify the process of scraping any `.pdf` documents, generating question-answer pairs using `openai`, and then uploading datasets 📊 to the Hugging Face Hub 🤗. It allows you to verify ✅, process 🔄, and push 🚀 your pandas DataFrame directly to Hugging Face, making it easier to share and collaborate 🤝 on datasets. In addition, new version allows users to fine-tune the Llama2 model over their proprietary data.

[![Watch the video](https://img.youtube.com/vi/XLExhyangWw/0.jpg)](https://www.youtube.com/watch?v=XLExhyangWw)

## Installation

To use **huggify-data**, ensure you have the necessary libraries installed. You can install them using pip:

```sh
pip install huggify-data
```

## Examples

Here's a complete example to illustrate how to use the **huggify-data** to scrape PDF and save as question-answer pairs in a `.csv` file. The block of code below will scrape it, convert it into a `.csv` and save the file locally.

```py
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

Here's a complete example to illustrate how to use the **huggify-data** library to push data (assuming an existing `.csv` file with columns `questions` and `answers` inside) to HuggingFace Hub:

```py
from huggify_data.push_modules import DataFrameUploader

# Example usage:
df = pd.read_csv('/content/toy_data.csv')
uploader = DataFrameUploader(df, hf_token="<huggingface-token-here>", repo_name='<desired-repo-name>', username='<your-username>')
uploader.process_data()
uploader.push_to_hub()
```

Here's a complete example to illustrate to use the **huggify-data** library to fine-tune a Llama-2 model (assuming you have a directory from HuggingFace existing):

```py
# Param
model_name = "NousResearch/Llama-2-7b-chat-hf" # recommended base model
dataset_name = "eagle0504/sample_toy_data_v9" # give a desired name, i.e. <hf_user_id>/<desired_name>
new_model = "youthless-homeless-shelter-web-scrape-dataset-v4" # give a desired name
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

To make an inference, please following the example below:

```py
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="eagle0504/youthless-homeless-shelter-web-scrape-dataset-v4") # same name as above
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