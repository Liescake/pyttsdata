import whisperx

device = "cuda"
audio_file = "test_audio1.mp3"
batch_size = 16
compute_type = "float16"
dir = "models"

model = whisperx.load_model("base", device, compute_type=compute_type, download_root=dir)

test_au1 = whisperx.load_audio(audio_file)
result = model.transcribe(test_au1, batch_size=batch_size)

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, test_au1, device, return_char_alignments=False)

print(result["segments"])