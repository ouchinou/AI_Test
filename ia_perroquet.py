import os
import queue
import sys
import time
import warnings

import numpy as np
import sounddevice as sd

# --- CONFIGURATION LOGS & ENVIRONNEMENT ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_NO_ADAPTATIVE_INT8"] = "1"

# Imports de nos modules locaux
from ai_engines import AIEngines
from audio_processor import AudioProcessor

# --- CONFIGURATION ---
FREQ_ECHANTILLONNAGE = 16000
BLOC_TAILLE = 1024
DELAI_SILENCE_SEC = 1.3
DOSSIER_DEBUG = "debug_audio"

file_audio = queue.Queue()


def callback_audio(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    file_audio.put(indata.copy())


def lancer_perroquet():
    # Instanciation de nos composants
    processor = AudioProcessor(sample_rate=FREQ_ECHANTILLONNAGE)
    engines = AIEngines()

    print("\n🎧 Chaîne audio filtrée prête.")
    print("✅ Le perroquet vous écoute. Parlez... (Ctrl+C pour quitter)")

    flux_micro = sd.InputStream(
        samplerate=FREQ_ECHANTILLONNAGE,
        channels=1,
        callback=callback_audio,
        blocksize=BLOC_TAILLE,
    )

    audio_phrase = []
    en_train_de_parler = False
    temps_dernier_son = time.time()

    with flux_micro:
        try:
            while True:
                bloc = file_audio.get()
                rms = processor.calculer_volume_rms(bloc)

                if rms > processor.noise_threshold:
                    if not en_train_de_parler:
                        print("\n🎙️ [Début de phrase détecté]")
                        en_train_de_parler = True
                    temps_dernier_son = time.time()

                if en_train_de_parler:
                    audio_phrase.append(bloc)

                    # Fin de phrase détectée
                    if time.time() - temps_dernier_son > DELAI_SILENCE_SEC:
                        print("🛑 [Fin de phrase détectée]")
                        en_train_de_parler = False

                        audio_brut = np.concatenate(audio_phrase, axis=0).flatten()
                        audio_phrase = []  # Reset

                        # Ignorer les bruits trop courts
                        if len(audio_brut) < FREQ_ECHANTILLONNAGE * 0.6:
                            continue

                        # Traitement DSP & Sauvegarde de test
                        audio_optimal = processor.nettoyer_signal(audio_brut)
                        nom_fichier = processor.sauvegarder_trace(
                            audio_optimal, DOSSIER_DEBUG
                        )

                        # 1. Transcription initiale via Whisper
                        texte_detecte = engines.transcrire(audio_optimal)

                        if texte_detecte:
                            print(f"📊 Tracé généré : {nom_fichier}")
                            print(f"📝 Transcription brute (Whisper) : {texte_detecte}")

                            # 2. Validation contextuelle et orthographique par Llama 3
                            print("🧠 Llama 3 valide et corrige les mots...")
                            texte_corrige = engines.reflechir_et_repondre(texte_detecte)
                            print(f"✨ Phrase finale validée : {texte_corrige}")

                            # 3. Synthèse vocale du texte validé
                            if texte_corrige:
                                flux_micro.stop()  # Coupe le micro pendant que l'IA parle
                                engines.synthetiser_et_jouer(texte_corrige)
                                flux_micro.start()  # Relance le micro
                            else:
                                print("⚠️ Abandon : Le texte validé est vide.")

                            print("\nÀ vous de parler...")
                            while not file_audio.empty():
                                file_audio.get_nowait()

        except KeyboardInterrupt:
            print("\nArrêt du perroquet.")


if __name__ == "__main__":
    lancer_perroquet()
