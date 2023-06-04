# EasyOCR-and-additional-function-for-text-from-image
## Introduction
This website has been developed to facilitate the extraction of text from images through the utilization of the EasyOCR library. Moreover, in addition to this capability, others APIs have been seamlessly integrated to extract higher-level meaning, including name entity recognition for detecting human names and text summarization employing the Huggingface framework.

## Technical Overview
**Streamlit** was utilized to quickly build the UI of the website.

**EasyOCR** was leveraged to **extract the English language** from the uploaded image since it provides faster results than Tesseract OCR with competitive accuracy.

For the task **name entity recognition**, **spacy library** provides efficient and accurate NER capabilities (spacy 3.5.3 release on May 15, 2023).

For the **summarization** task, the standard summarization pipeline on **HuggingFace** was prepared for that task (sshleifer/distilbart-cnn-12-6).

Advanced setting: 
**GPU** could be utilized for faster extract text from images, and the summarization process. Which will be activated when the GPU is available on your machine through the check command (if torch.cuda.is_available()).

Since the limitation of our account, the **OpenAI API chatbox** was added as a future expanding. You should set "YOUR_OPENAI_API_KEY" for the variable "openAI_API_key" on the source code to activate this function.

## How to Install
version of pip: 23.0.1, 23.1.2
### On Windows
python -m venv myenv

myenv\Scripts\activate

pip install -r requirements.txt

### On Mac
python3 -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

## How to run
Activate the virtual environment before running

streamlit run main.py
