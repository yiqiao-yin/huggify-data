# huggify-data

## Introduction

**huggify-data** 📦 is a Python library 🐍 designed to simplify the process of scraping any `.pdf` documents, generating question-answer pairs using `openai`, and then uploading datasets 📊 to the Hugging Face Hub 🤗. It allows you to verify ✅, process 🔄, and push 🚀 your pandas DataFrame directly to Hugging Face, making it easier to share and collaborate 🤝 on datasets.

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
openai_api_key = "sk-API_KEY_HERE
generator = PDFQnAGenerator(pdf_path, openai_api_key)
generator.process_scraped_content()
generator.generate_questions_answers()
df = generator.convert_to_dataframe()
print(df)
```

Here's a complete example to illustrate how to use the **huggify-data** library:

```py
from huggify_data.push_modules import DataFrameUploader

# Example usage:
df = pd.read_csv('/content/toy_data.csv')
uploader = DataFrameUploader(df, hf_token="<huggingface-token-here>", repo_name='<desired-repo-name>', username='<your-username>')
uploader.process_data()
uploader.push_to_hub()
```

## Class Details

### DataFrameUploader

**DataFrameUploader** is the main class provided by **huggify-data**. 

#### Initialization

```py
uploader = DataFrameUploader(df, hf_token="<huggingface-token-here>", repo_name='<desired-repo-name>', username='<your-username>')
```

- **df**: A pandas DataFrame containing the data.
- **hf_token**: Your Hugging Face API token.
- **repo_name**: The desired name for the Hugging Face repository.
- **username**: Your Hugging Face username.

#### Methods

- **verify_dataframe()**:
    - Checks if the DataFrame has columns named `questions` and `answers`.
    - Raises a `ValueError` if the columns are not present.
  
- **process_data()**:
    - Verifies the DataFrame.
    - Converts the data into a DatasetDict object.

- **push_to_hub()**:
    - Creates a repository on the Hugging Face Hub.
    - Pushes the DatasetDict to the repository.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/yiqiao-yin/huggify-data/blob/main/LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

## Contact

For any questions or support, please contact [eagle0504@gmail.com](mailto: eagle0504@gmail.com).