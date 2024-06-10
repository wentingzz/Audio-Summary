import gradio as gr
import requests
import os
import torch
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model


#######------------- LLM-------------####

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}
params = {
        GenParams.MAX_NEW_TOKENS: 800, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }
LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )
llm = WatsonxLLM(LLAMA2_model)  

#######------------- Prompt Template-------------####

temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""
pt = PromptTemplate(
    input_variables=["context"],
    template= temp)
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2text-------------####

def transcribe_url(url, downloaded_audio_path="temp.mp3"):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(downloaded_audio_path, 'wb') as file:
            file.write(response.content)
        print("File downloaded")
        res = transcript_audio(downloaded_audio_path)
        try:
            os.remove(downloaded_audio_path)
            print("Temporary file deleted")
        except OSError as e:
            print(f"Error: {downloaded_audio_path} : {e.strerror}")
        return res
    else:
        print("Failed to download the file")
        return "Failed to download the file"

def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    summary = prompt_to_LLAMA2.run(transcript_txt) 
    return summary

#######------------- Gradio-------------####

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output
iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, outputs=output_text, 
    title="Audio Summary",
    description="Upload an audio file in MP3 format and get a summarized text of its content. This app uses advanced speech recognition and summarization models to condense spoken words into a concise summary. Simply choose your audio file and let the app process and display the summary for you."
)
# Adding URL input
url_input = gr.Textbox(label="Audio URL", placeholder="Enter URL to an audio file")
url_output = gr.Textbox(label="Transcription")

# Combine interfaces
iface_url = gr.Interface(
    fn=transcribe_url, 
    inputs=url_input, outputs=url_output,
    title="URL-Based Audio Summary",
    description="Enter the URL of an audio file to get a summarized text of its content. This app downloads the audio from the provided URL and uses advanced speech recognition and summarization models to create a concise summary. Just paste the URL and receive the summary in seconds."
)
iface_combined = gr.TabbedInterface(
    interface_list=[iface, iface_url], 
    tab_names=["Upload Audio", "URL Audio"],
    title="Audio Summary App"
)

# Launch the interface
iface_combined.launch(server_name="0.0.0.0", server_port=8080)
