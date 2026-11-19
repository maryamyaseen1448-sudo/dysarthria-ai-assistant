# Dysarthria Assistant — Working Research Prototype

A local web application based on the proposal's workflow:
Speech Input → Speech Recognition → Personalized AI → Confidence Check → Clarification → Clear Text/Voice Output.

## Requirements
- Python 3.10 or newer
- Internet for the first installation/model download
- Chrome or Edge recommended for microphone recording
- Microphone permission for live recording

## Run on Windows
1. Extract this ZIP.
2. Open Command Prompt in the extracted `dysarthria_communication_interface` folder.
3. Create and activate a virtual environment (recommended):
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bat
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bat
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in Chrome or Edge.

The first run downloads the Whisper `small` model and may take time. To use a smaller model:
```bat
set WHISPER_MODEL=base
python app.py
```

## Connect a Kaggle dysarthric speech dataset
1. Download the dataset from Kaggle and extract it.
2. Copy audio files (WAV, MP3, M4A, FLAC, OGG, WEBM) into the project's `dataset` folder, keeping subfolders if useful.
3. Refresh the Dataset Sample list in the app.
4. Choose a sample and press Process Speech.

Alternatively, set `DYSARTHRIA_DATASET_DIR` to the extracted dataset folder before starting the app:
```bat
set DYSARTHRIA_DATASET_DIR=C:\path\to\your\kaggle_dataset
python app.py
```
This prototype discovers audio files recursively. It does not automatically parse every Kaggle dataset's CSV/transcript format; dataset-specific transcript/speaker metadata mapping should be added after the exact Kaggle dataset is selected.

## What works
- Browser microphone recording (MediaRecorder)
- Audio file upload and playback
- Dataset audio sample browsing from a local extracted folder
- Local speech-to-text with faster-whisper
- User-defined personal correction pairs applied to future transcripts
- Estimated confidence indicator and low-confidence clarification prompt
- User confirmation/editing of the intended message
- Text-to-speech using the browser's speech synthesis
- Copy final text

## Research and evaluation notes
- The confidence indicator is a heuristic based on Whisper segment log-probability and no-speech probability. It is not a calibrated probability, medical measure, or clinical validation.
- Personalization in this first prototype is a user-maintained correction dictionary, not a trained speaker-specific acoustic model.
- Evaluation with a Kaggle dataset should compare predicted transcripts with the dataset's reference transcripts using WER/CER and report results by speaker where metadata permits.
- No live participant study is built in; the proposal describes initial technical evaluation using an existing dataset.
- This is a research prototype, not a diagnostic or treatment tool.

## Project structure
- `app.py` — Flask backend and local speech recognition
- `templates/index.html` — responsive interface and browser interactions
- `dataset/` — place extracted dataset audio here
- `requirements.txt` — Python dependencies
