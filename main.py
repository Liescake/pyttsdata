import json
import os
from pathlib import Path
from typing import Any, List, Dict

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

def locate_audio(deepth=1, targets: List = None, extensions=None):
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

def transcribe(conf=None, audios: List[Path] = None):
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
    results = []
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
        results.append(result)
        print(results)

    return results
        
if __name__ == "__main__":
    load_config()
    audios = locate_audio()
    if audios:
        transcribe(audios=audios)
