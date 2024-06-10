# Audio Summary

The Audio Summary project provides a tool to generate summaries for uploaded audio files. This can be useful for quickly understanding the content of a long audio recording or interview.

## Features

- **Audio Upload**: Users can upload audio files in various formats.
- **Text Summary**: The system generates a text summary of the audio content.
- **Playback Support**: Users can playback the uploaded audio for reference.

## Installation

Clone the repository:

```bash
git clone https://github.com/wentingzz/Audio-Summary.git
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the application:
```bash
python audio_summary_openai_ibm.py
```
Or
```bash
python3 audio_summary_openai_ibm.py
```

2. Open the application in your web browser.
3. Upload an audio file or provide the URL of audio file.
4. Click the "Submit" button. The system will process the audio and display a text summary.

## Technologies Used

- **Transformers**: For natural language processing tasks.
- **Torch**: Deep learning framework for PyTorch.
- **Gradio**: For building the web interface.
- **Langchain**: A library for building and managing language model pipelines.
- **IBM Watson Machine Learning**: For model deployment and management.
- **Hugging Face Hub**: For accessing pre-trained models.
