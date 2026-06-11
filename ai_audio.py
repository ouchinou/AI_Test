from faster_whisper import WhisperModel


def transcrire(fichier):
    # UV installe CUDA par défaut, on en profite !
    model = WhisperModel("small", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(fichier, language="fr")
    return " ".join([seg.text for seg in segments])


if __name__ == "__main__":
    # print(transcrire("audio.mp3"))
    pass
