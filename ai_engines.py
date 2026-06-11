import ollama  # Émanation de votre dépendance installée
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline


class AIEngines:
    def __init__(self, modele_llm="llama3"):
        print("🤖 Initialisation de Whisper (GPU - CUDA 12.6)...")
        self.whisper = WhisperModel("small", device="cuda", compute_type="int8_float16")

        print("🗣️ Initialisation de Kokoro (GPU - CUDA 12.6)...")
        self.kokoro = KPipeline(
            lang_code="f", repo_id="hexgrad/Kokoro-82M", device="cuda"
        )

        # Configuration d'Ollama
        self.modele_llm = modele_llm
        print(f"🧠 Module de langage configuré sur Ollama ({self.modele_llm})...")

    def transcrire(self, audio_data):
        """Convertit le signal audio nettoyé en texte et filtre les hallucinations."""
        segments, _ = self.whisper.transcribe(audio_data, beam_size=3, language="fr")
        texte = "".join([seg.text for seg in segments]).strip()

        # Filtre anti-hallucination
        phrases_parasites = [
            "Sous-titres réalisés par la communauté d'Amara.org",
            "Amara.org",
            "Merci d'avoir regardé cette vidéo !",
        ]
        for parasite in phrases_parasites:
            texte = texte.replace(parasite, "")
        return texte.strip()

    def reflechir_et_repondre(self, texte_utilisateur):
        try:
            consigne_correcteur = (
                "Tu es un correcteur orthographique et grammatical pour de la reconnaissance vocale. "
                "Ton unique rôle est de corriger les erreurs de transcription. "
                "REGLES STRICTES :\n"
                "1. Ne réponds JAMAIS à la phrase.\n"
                "2. Renvoie UNIQUEMENT la phrase corrigée, sans commentaires ni introduction.\n"
                "3. Conserve le sens exact de ce que l'utilisateur a dit."
            )
            reponse = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": consigne_correcteur},
                    {"role": "user", "content": f"Corrige : {texte_utilisateur}"},
                ],
            )
            return reponse["message"]["content"].strip()
        except Exception:
            return texte_utilisateur

    def synthetiser_et_jouer(self, texte, voix="ff_siwis", vitesse=1.0):
        """Génère la voix artificielle et bloque la lecture jusqu'à la fin."""
        generator = self.kokoro(texte, voice=voix, speed=vitesse)
        for _, _, audio_synthese in generator:
            audio_numpy = audio_synthese.cpu().numpy()
            sd.play(audio_numpy, samplerate=24000)
            sd.wait()
