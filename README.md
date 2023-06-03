# EasyOCR-and-additional-function-for-text-from-image
## Introduction
This website has been developed to facilitate the extraction of text from images through the utilization of the EasyOCR library. Moreover, in addition to this capability, various other APIs have been seamlessly integrated to extract higher-level meaning, including name entity recognition for detecting human names and text summarization employing the Huggingface framework.

## Technical Overview
Streamlit was utilized to quickly build the UI of website.
EasyOCR was leveraged to extract the english language from upload image, since it provide faster results than Tesseract OCR with competitive accuracy.
For the task name entity recognition, it provide efficient and accurate NER capabilities (spacy 3.5.3 release on May 15, 2023).
For summarization task, the standard summarization pipeline on HuggingFace was prepered for that task (sshleifer/distilbart-cnn-12-6).

Advanced setting: 
GPU could be utilzied for faster extract text from image, and the summarization process. Which will be activated when the GPU is avaiable on your machine through the check command (if torch.cuda.is_available()).
Since the limited of our account, OpenAI API chatbox was added as an future expanding. You should set "YOUR_OPENAI_API_KEY" for the variable "openAI_API_key" on the source code.

## How to Install
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
