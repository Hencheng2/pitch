import os
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_pitch_shift(audio_path, start_time, end_time, semitones):
    """
    Apply pitch shift to a specific portion of the audio.
    semitones: positive = higher pitch, negative = deeper voice
    """
    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=None)
    
    # Convert times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract the portion to modify
    portion = y[start_sample:end_sample]
    
    # Apply pitch shift to the portion
    shifted_portion = librosa.effects.pitch_shift(portion, sr=sr, n_steps=semitones)
    
    # Reconstruct the audio
    result = np.concatenate([y[:start_sample], shifted_portion, y[end_sample:]])
    
    # Save to temporary file
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.wav")
    sf.write(output_path, result, sr)
    
    return output_path

def apply_pitch_shift_pydub(audio_path, start_time, end_time, semitones):
    """
    Alternative method using pydub for better format support
    """
    audio = AudioSegment.from_file(audio_path)
    
    # Convert to milliseconds
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)
    
    # Split the audio
    before = audio[:start_ms]
    portion = audio[start_ms:end_ms]
    after = audio[end_ms:]
    
    # Apply pitch shift using speed change (pitch shift via sample rate change)
    # semitones = 12 * log2(speed_factor)
    speed_factor = 2 ** (semitones / 12)
    new_sample_rate = int(portion.frame_rate * speed_factor)
    shifted_portion = portion._spawn(portion.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(portion.frame_rate)
    
    # Combine all parts
    result = before + shifted_portion + after
    
    # Save to temporary file
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.wav")
    result.export(output_path, format="wav")
    
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{uuid.uuid4().hex}_{filename}")
        file.save(input_path)
        
        # Get processing parameters
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', 0))
        intensity = float(request.form.get('intensity', -5))  # negative = deeper
        
        # Apply pitch shift
        try:
            output_path = apply_pitch_shift_pydub(input_path, start_time, end_time, intensity)
            
            # Clean up input file
            os.remove(input_path)
            
            # Return the processed file
            return send_file(
                output_path,
                as_attachment=True,
                download_name='processed_audio.wav',
                mimetype='audio/wav'
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/process-from-recording', methods=['POST'])
def process_recording():
    data = request.json
    audio_base64 = data.get('audio')
    start_time = float(data.get('start_time', 0))
    end_time = float(data.get('end_time', 0))
    intensity = float(data.get('intensity', -5))
    
    if not audio_base64:
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Decode base64 audio
    audio_data = base64.b64decode(audio_base64.split(',')[1])
    
    # Save to temporary file
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"recording_{uuid.uuid4().hex}.webm")
    with open(input_path, 'wb') as f:
        f.write(audio_data)
    
    # Apply pitch shift
    try:
        output_path = apply_pitch_shift_pydub(input_path, start_time, end_time, intensity)
        
        # Clean up input file
        os.remove(input_path)
        
        # Return the processed file
        return send_file(
            output_path,
            as_attachment=True,
            download_name='processed_recording.wav',
            mimetype='audio/wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
