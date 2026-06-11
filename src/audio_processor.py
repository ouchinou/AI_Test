import os

import noisereduce as nr
import numpy as np
import scipy.io.wavfile as wav


class AudioProcessor:
    def __init__(self, sample_rate=16000, noise_threshold=0.015):
        self.sample_rate = sample_rate
        # On redescend le seuil de détection car l'audio sera nettoyé proprement
        self.noise_threshold = noise_threshold

    def nettoyer_signal(self, audio_brut):
        """
        Utilise l'algorithme NoiseReduce pour soustraire le bruit de fond
        sans hacher ni robotiser la voix humaine.
        """
        if len(audio_brut) == 0:
            return audio_brut

        try:
            # 1. Réduction algorithmique du bruit de fond (méthode adaptative)
            audio_nettoye = nr.reduce_noise(
                y=audio_brut,
                sr=self.sample_rate,
                prop_decrease=0.95,  # Réduction à 95% du bruit détecté
                stationary=True,  # Idéal pour les bruits de soufflerie/micro continu
            )

            # 2. Normalisation du gain pour Whisper
            max_amplitude = np.max(np.abs(audio_nettoye))
            if max_amplitude > 0.01:
                return audio_nettoye * (0.8 / max_amplitude)

            return audio_nettoye

        except Exception as e:
            print(f"⚠️ Échec du nettoyage IA : {e}. Retour au signal brut.")
            return audio_brut

    def calculer_volume_rms(self, bloc_audio):
        """Calcule la puissance d'un bloc pour la VAD."""
        return np.sqrt(np.mean(bloc_audio**2))

    def sauvegarder_trace(self, audio_data, dossier_debug):
        """Sauvegarde le fichier WAV pour analyse QA."""
        from datetime import datetime

        os.makedirs(dossier_debug, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nom_fichier = os.path.join(dossier_debug, f"phrase_isolee_{timestamp}.wav")
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav.write(nom_fichier, self.sample_rate, audio_int16)
        return nom_fichier
