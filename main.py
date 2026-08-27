import json
import os
from pathlib import Path
from typing import Any, List, Dict
from typing import Optional


def load_config(flag=4):
    # File mode
    if flag == 1:
        pass
    # Dir mode
    elif flag == 2:
        pass
    # Mutiple mode
    elif flag == 3:
        pass
    # Config file mode
    elif flag == 4:
        with open("config.json", encoding="utf-8") as config_file:
            global config
            config = json.load(config_file)
    else:
        raise Exception("UnknownTaskType")

def locate_audio(targets: Optional[List] = None, extensions=None):
    # 为了适配后续终端操作，所以targets作为参数传入
    settings = config["whisperx"]
    if targets is None:
        targets = []
    if settings["targets"] == [] and not targets:
        targets = [Path(settings["default_target"])]
    elif settings["targets"] and not targets:
        for i in settings["targets"]:
            p = Path(i)
            if p.exists():
                targets.append(p)
    if extensions is None:
        extensions = settings["default_extensions"]
        
    result = []
    
    # TODO: Add dir walk deepth arg and logic
    for t in targets:
        if t.is_dir():
            for current_dir, dirnames, filenames in t.walk():
                for fn in filenames:
                    _, ext = os.path.splitext(fn)
                    if ext.lower() in extensions:
                        result.append(current_dir / fn)
        elif t.is_file():
            _, ext = os.path.splitext(t.name)
            if ext in extensions:
                result.append(t)

    return result

def transcribe(conf=None, audios: Optional[List[Path]] = None):
    import copy
    if conf is None:
        load_config()
        conf = copy.deepcopy(config)
    if audios is None:
        audios = locate_audio()

    import whisperx
    settings = conf["whisperx"]
    model = whisperx.load_model(
        settings["model"],
        settings["device"],
        compute_type=settings["compute_type"],
        download_root=settings["download_root"],
    )
    results = {}
    for audio_path in audios:
        audio = whisperx.load_audio(str(Path(audio_path).resolve()))
        result = model.transcribe(audio, batch_size=settings["batch_size"])
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=settings["device"],
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            settings["device"],
            return_char_alignments=False,
        )
        for segment in result.get("segments", []):
            segment.pop("words", None)
            segment.pop("avg_logprob", None)
        result.pop("word_segments", None)
        
        results[audio_path] = result.get("segments", [])
        
    print(results)
    return results
        
def timestamp_convert(input: Dict[Path, List[Dict[str, Any]]]):
    def format_timestamp(seconds: Any) -> str:
        total_seconds = int(float(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    for segments in input.values():
        for segment in segments:
            segment["start"] = format_timestamp(segment["start"])
            segment["end"] = format_timestamp(segment["end"])

    return input

    

def split(targets: Dict[Path, List[Dict[str, Any]]], conf: Any = None):
    import ffmpeg

    t = {
        "audio": "",
        "text": "",
        "ref_audio": ""
    }

    task_sequence = []
    counter = 0
    
    if conf == None: conf = config
    output_dir = Path(conf["ffmpeg"]["opt_file"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir.parent / "train_raw.txt", "a", encoding="utf-8") as f:
            
        for a, l in targets.items():
            for i in l:
                print(i)
                
                output_path = output_dir / f"{counter}.wav" if counter != 0 else output_dir / f"ref_audio.wav"
                task_sequence.append(
                    ffmpeg.input(str(a))
                    .audio
                    .filter("atrim", start=i["start"], end=i["end"])
                    .filter("asetpts", "PTS-STARTPTS")
                    .output(str(output_path), format="wav", acodec="pcm_s16le")
                )
                
                t["audio"] = str(output_path)
                t["text"] = i["text"]
                t["ref_audio"] = str(output_path)
                
                f.write(json.dumps(t))
                f.write("\n")

                counter += 1
                
    for task in task_sequence:
        task.run(overwrite_output=True)

    
if __name__ == "__main__":
    load_config()
    audios = locate_audio()
    if audios:
        split(transcribe(audios=audios))
