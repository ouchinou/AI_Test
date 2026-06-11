import os
import warnings

import sounddevice as sd
import torch
from kokoro import KPipeline

# 1. Masquer les avertissements de Windows/PyTorch
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Masquer l'avertissement spécifique aux Symlinks de Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 3. Masquer le message d'avertissement d'authentification (Optionnel)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def parler_francais(texte, voix="ff_siwis"):
    """
    Génère de la parole en français et la joue.
    Voix françaises disponibles :
    - 'ff_siwis' (Femme - excellente qualité)
    - 'fr_viona' (Femme)
    - 'fr_gilles' (Homme)
    """
    print("Chargement automatique du modèle Kokoro (exécution sur GPU)...")

    # On initialise le pipeline pour le français ('f')
    # et on force l'utilisation de votre carte graphique NVIDIA (cuda)
    pipeline = KPipeline(
        lang_code="f", device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Génération de la voix pour : '{texte}'")

    # Le pipeline génère des blocs de texte/audio
    generator = pipeline(texte, voice=voix, speed=1.0)

    print("Lecture audio...")
    for graphemes, phonemes, audio in generator:
        # audio est un tenseur PyTorch, on le convertit en tableau numpy pour sounddevice
        audio_numpy = audio.cpu().numpy()

        # Kokoro génère nativement en 24000 Hz
        sd.play(audio_numpy, samplerate=24000)
        sd.wait()

    print("Lecture terminée.")


if __name__ == "__main__":
    texte_test = "Bonjour ! Je tourne entièrement en local sur votre ordinateur Lenovo grâce à votre carte graphique. La qualité est plutôt naturelle, non ?"
    parler_francais(texte_test)
