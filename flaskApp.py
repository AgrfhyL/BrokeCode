from flask import Flask, request, jsonify
import torch
import librosa
import io
import soundfile as sf
import tempfile
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Configure device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load your processor and model ---
# Replace the paths below with the appropriate local paths or model identifiers.
processor = Wav2Vec2Processor.from_pretrained("/home/agrfhyl/processor")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# Load your model state dictionary if you have a custom-trained 
state_dict = torch.load("/home/agrfhyl/model_state_dict.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # set model to evaluation mode

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    try:
        # Save the uploaded file temporarily
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False)
        temp_audio_path.write(audio_bytes)
        temp_audio_path.close()

        # Use soundfile to load the audio (assuming it's already 16kHz)
        audio_array, sr = sf.read(temp_audio_path.name)

        # Ensure the sample rate is 16kHz (if it's not, you can resample it here)
        if sr != 16000:
            raise ValueError("Audio sample rate is not 16kHz")

    except Exception as e:
        return jsonify({"error": f"Could not process audio: {str(e)}"}), 500

    # Preprocess the audio for the model
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    # Perform transcription using the model
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs.logits

    # Decode the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Cleanup temporary files
    os.remove(temp_audio_path.name)

    return jsonify({"transcription": transcription})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
