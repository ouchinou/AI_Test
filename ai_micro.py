import queue
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Configuration audio
FREQ_ECHANTILLONNAGE = 16000  # Whisper travaille nativement en 16kHz
DUREE_BLOC = 3  # Analyse l'audio par blocs de 3 secondes pour rester réactif

# File d'attente pour stocker les morceaux d'audio du micro
file_audio = queue.Queue()


def callback_audio(indata, frames, time, status):
    """Cette fonction est appelée automatiquement par sounddevice à chaque fois que le micro reçoit du son."""
    if status:
        print(status, file=sys.stderr)
    # On ajoute une copie du bloc audio dans notre file d'attente
    file_audio.put(indata.copy())


def ecouter_et_transcrire():
    # 1. Charger Whisper sur votre carte graphique RTX 2000
    print("Initialisation de Whisper sur le GPU...")
    model = WhisperModel("small", device="cuda", compute_type="float16")

    print(f"\n🎤 Microphone activé (Analyse toutes les {DUREE_BLOC} secondes).")
    print("Parlez... (Appuyez sur Ctrl+C dans le terminal pour arrêter)")

    # 2. Ouvrir le flux du microphone en entrée
    flux_micro = sd.InputStream(
        samplerate=FREQ_ECHANTILLONNAGE, channels=1, callback=callback_audio  # Mono
    )

    audio_accumule = np.zeros((0, 1), dtype=np.float32)

    with flux_micro:
        try:
            while True:
                # Récupérer le morceau d'audio disponible dans la file d'attente
                # timeout=None bloque la boucle tant qu'il n'y a pas de son
                bloc = file_audio.get()
                audio_accumule = np.vstack((audio_accumule, bloc))

                # Dès qu'on a accumulé assez de secondes d'audio, on l'analyse
                if len(audio_accumule) >= FREQ_ECHANTILLONNAGE * DUREE_BLOC:
                    # Whisper attend un tableau de float32 aplati (1D)
                    audio_data = audio_accumule.flatten()

                    # Lancer la transcription
                    # beam_size=3 est un bon compromis vitesse/précision pour le direct
                    segments, info = model.transcribe(
                        audio_data, beam_size=3, language="fr"
                    )

                    # Récupérer et afficher le texte
                    texte = "".join([segment.text for segment in segments]).strip()

                    if texte:
                        print(f"👉 {texte}")

                    # Réinitialiser pour le prochain bloc
                    audio_accumule = np.zeros((0, 1), dtype=np.float32)

        except KeyboardInterrupt:
            print("\nArrêt de l'écoute micro.")


if __name__ == "__main__":
    ecouter_et_transcrire()
