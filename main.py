import argparse
import numpy as np
import streamlit as st
from PIL import Image
import torch
import time
import easyocr
from PIL import Image
import cv2
import spacy
from transformers import pipeline
import pdfkit
import subprocess
import openai


@st.cache_resource        
def define_openai_key(openAI_API_key):
    openai.api_key = openAI_API_key


# this class was built to store the PDF Configuration
class PDFConfig():
    def __init__(self, font_size=20, color='Black') -> None:
        """
         Initialize the color and font size. This is the constructor for the PDF Configuration class
         
         @param font_size - The font size to use
         @param color - The color to use (default is 'Black')
         
        """
        self.font_size = font_size
        self.color = color


@st.cache_resource        
def define_global_variables():
    """
     Define variables that need to be defined before any pipeline is run. This is a helper function to be called by the pipeline_init function.
     
     @return device, len_clip_feature, ner, text_infor, ocr_reader, summarizer, PDFConfig(), web_state
    """
    print('Define global variables')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    len_clip_feature = 512
    ner = spacy.load("en_core_web_sm")
    text_infor = {}
    if device == 'cpu':
        ocr_reader = easyocr.Reader(['en'], gpu = False)
        summarizer = pipeline("summarization", device=-1)
    else:
        ocr_reader = easyocr.Reader(['en'], gpu = True)
        summarizer = pipeline("summarization", device=0)
    if openAI_API_key != "YOUR_OPENAI_API_KEY":
        define_openai_key(openAI_API_key)
    return device, len_clip_feature, ner, text_infor, ocr_reader, summarizer, PDFConfig(), {}


global args
global device
global len_clip_feature 
global ner
global text_infor
global ocr_reader
global summarizer
global pdfConfig
global web_state
global openAI_API_key
openAI_API_key = "YOUR_OPENAI_API_KEY"
device, len_clip_feature, ner, text_infor, ocr_reader, summarizer, pdfConfig, web_state = define_global_variables()


def get_text_from_image(img, ocr_reader, threshold=None, withBoudingBox=False):
    """
     Get text from image using OCR technique.
     
     @param img - image to get text from
     @param ocr_reader - reader used to read OCR
     @param threshold - if set only return text with a value greater than this threshold
     @param withBoudingBox - if True return the image with bounding box
    """
    result = ocr_reader.readtext(img, batch_size=32)
    result_with_threshold = []

    # Add threshold to result if threshold is not None
    if threshold is not None:
        for item in result: 
            if item[2] > 0.5:
                result_with_threshold.append(item)

    # Returns OCR result and image with bounding box
    if withBoudingBox:
        # Draw a rectangle of the image
        for item in result:
            point1, point2, point3, point4 = item[0]
            point1, point3 = list(map(int, point1)), list(map(int, point3))
            img = cv2.rectangle(img, (point1), (point3), (0, 255, 0), 2)
        return result_with_threshold, img
    return result_with_threshold


def convert_ocr_result_to_paragraph(ocr_result):
    """
     Convert all text was detected by EasyOCR to a paragraph
     
     @param ocr_result - A list of EasyOCR result
     
     @return A string containing the result of ocr_text concatenation
    """
    result = ''
    for item in ocr_result:
        result += ' ' + (str)(item[1])
    result = result[1:]
    return result


def get_person_from_paragraph(paragraph, ner):
    """
     Get person from paragraph. This is a helper function to detect the person name in a paragraph.
     
     @param paragraph - paragraph to get person from. It should be a string.
     @param ner - NER class to use for parsing.
     
     @return list of person's names as strings.
    """
    result = []
    doc = ner(paragraph)
    for entity in doc.ents:
      if entity.label_ == 'PERSON':
        result.append(entity.text)
    return result


def generate_text_for_pdf(paragraph, pdfConfig):
    """
     This is a helper function to generate text to PDF file.
     
     @param paragraph - The text to be convert. It should be a string.
     @param pdfConfig - The configuration of PDF.
     
     @return The text was generated for pdfkit library to convert to PDF file.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {font-size: {font_size}; color:{color}};}
        </style>
    </head>
    <body>
        <div>{content}</div>
    </body>
    </html>
    """

    content_strings = paragraph.split('<br>')
    content_insert = ''
    for content_string in content_strings:
        content_insert += '<div>' + content_string + '</div><br>'
    html_content = html_template.replace('<div>{content}</div>', content_insert)
    html_content = html_content.replace('{font_size}', (str)(pdfConfig.font_size) + 'px')
    html_content = html_content.replace('{color}', pdfConfig.color)
    return html_content


def generate_response(prompt):
    """
     Generate a response to the user. It is used as a chatbox for customer question.
     
     @param prompt - The question from the user
     
     @return The OpenAI's response to the user
    """
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=None
    # stop=[" Human:", " AI:"]
    )
    return response.choices[0].text.strip()


def streamlit_app():
    define_global_variables()
    st.markdown("""
    <style>
        .title {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Easy OCR, NER and summarization for text analysis from image</h1>", unsafe_allow_html=True)
    st.subheader('Duong Van Khoa - dvkhoa19@apcs.fitus.edu.vn')
    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox("Select Page", ["Home", "About", "Help"])

    if section == "Home":
        uploaded_file = st.file_uploader("Choose an image to upload!!!")
    
        # upload image
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img = img.convert("RGB")
            st.image(img, caption='Uploaded Image', use_column_width=True)
        else:
            web_state['image_with_ocr'] = None
            text_infor['Exist'] = False
            
        # process image (detect content from image, detect human name, text summarization)
        if st.button("Start the text extraction process"):
            if 'img' not in locals() or 'img' == None:
                st.write('You have to upload the image file before process!')
            else:
                start = time.time()
                img = np.array(img)
                results, img = get_text_from_image(img, ocr_reader=ocr_reader, threshold=0.5, withBoudingBox=True)
                web_state['image_with_ocr'] = np.copy(img)
                text_infor['paragraph'] = convert_ocr_result_to_paragraph(results)
                person_names = get_person_from_paragraph(text_infor['paragraph'], ner)
                summary = summarizer(text_infor['paragraph'], do_sample=False) 
                text_infor['summary'] = summary[0]['summary_text']
                end = time.time()
                web_state['query_time'] = end - start

                text_infor['Person name'] = ''
                for person_name in person_names:
                    text_infor['Person name'] += person_name + ', '
                if text_infor['Person name'] != '':
                    text_infor['Person name'] = text_infor['Person name'][:-2]
                text_infor['Exist'] = True
                

        # show the image with bounding box for text detection
        #   and some additionnal UI function (download PDF, custom text summarization)
        if len(text_infor) != 0 and 'img' in locals() and 'img' != None and text_infor['Exist']:
            if web_state['image_with_ocr'] is not None:
                st.image(web_state['image_with_ocr'], caption='Image with text bounding box')
                st.write('**Detect words:** ' + text_infor['paragraph'])
                st.write('**Names of people in text:** ' + text_infor['Person name'])
                st.write('**Summary of text:** ' + text_infor['summary'])
                tmp_query_time = web_state['query_time']
                st.write(f'**Processing time:** {tmp_query_time:.3f}s')
            if 'Person name' in text_infor and 'summary' in text_infor:
                pdf_string = 'Detect words: ' + text_infor['paragraph'] + '<br>' + 'Names of people in text: ' + text_infor['Person name'] + '<br>' + 'Summary of text: ' + text_infor['summary']
                config = pdfkit.configuration(wkhtmltopdf='C:\\Users\\duong\\Downloads\\Cinnamon\\mvenv\\Scripts\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
                with st.expander("PDF Download and Configuration"):
                    value = st.slider("Select font size", min_value=20, max_value=35, value=pdfConfig.font_size, step=1)
                    pdfConfig.font_size = value
                    color_options = ["Tomato", "DodgerBlue", "MediumSeaGreen", "Black"]
                    color = st.selectbox("Select color", (color_options), index=color_options.index(pdfConfig.color))
                    pdfConfig.color = color
               
                    html_content = generate_text_for_pdf(pdf_string, pdfConfig)
                    pdfkit.from_string(html_content, 'out.pdf', configuration=config, options={"enable-local-file-access": ""})
                    with open('out.pdf', 'rb') as f:
                        download_button = st.download_button(label="Download PDF", data=f, file_name="output.pdf")

                with st.expander("Custom text for summarization"):
                    text_input = st.text_input('Input text for summarization')
                    if st.button("Process"):
                        max_length = len((str)(text_input).split(' '))
                        min_length = (int)(max_length / 2)
                        summary = summarizer(text_input, max_length=max_length, min_length=min_length, do_sample=False) 
                        summary = summary[0]['summary_text']
                        st.write('Summary: ' + summary)

                with st.expander("Using ChatGPT box"):
                    if openAI_API_key != "YOUR_OPENAI_API_KEY":
                        user_input = st.text_input("User Input", "")
                        if st.button("Generate Response"):
                            response = generate_response(user_input)
                            st.text_area("AI Response", value=response)
                    else:
                        st.write("Your have to set the API key for openAI_API_key variable to use this function!")
                        st.write("Since it requires a fee for the API, currently I don't activate this function!")

                
    elif section == "About":
        st.title("About")
        st.write("This website has been developed to facilitate the extraction of text from images through the utilization of the EasyOCR library. Moreover, in addition to this capability, various other APIs have been seamlessly integrated to extract higher-level meaning, including name entity recognition for detecting human names and text summarization employing the Hugginface framework.")
        st.write("Thanks to my application in the **Cinnamon AI Bootcamp**, which motivated me for constructing this website from scratch with the support from Streamlit library for fast UI building. This endeavor was made to represent my AI research and engineering skills to deliver **a quick demo application**.")
        st.write("My gmail: dvkhoa19@apcs.fitus.edu.vn")
        st.write("My phone number: 0343623811")
        st.write("Faculty of Information Technology")
        st.write("University of Science")

    elif section == "Help":
        st.title("Help")
        st.write("The expander **\'Custom text for summarization\'** was added to support the summarization process. Since the raw summarization after text processing gets the content from concatenating all results of EasyOCR, which could be not appropriate for specific cases. You can utilize this expander for summarizing the specific paragraph.")
        st.write("The expander **\'PDF Download and Configuration\'**  act as an interface to config some pdf format and download file. Currently, color and size was two configurations, that you are free to choose from.")
        st.write("You can leverage additional tools to gain a deeper utilize of the text extraction: https://chat.openai.com/, https://bard.google.com/?hl=en")

if __name__ == '__main__':
    streamlit_app()  