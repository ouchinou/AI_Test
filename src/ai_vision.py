import cv2  # Pour la capture vidéo
from ultralytics import YOLO  # Pour l'IA de détection


def detecter_en_temps_reel():
    # 1. Charger le modèle YOLO (ici le petit modèle "nano", très rapide) sur le GPU (device=0)
    # Pour la vidéo, la vitesse est cruciale.
    print("Initialisation du modèle sur le GPU RTX...")
    model = YOLO("yolov8n.pt")

    # 2. Ouvrir la webcam par défaut (index 0 sur la plupart des PC portables)
    print("Ouverture de la webcam (entrée)...")
    # Sur certains PC, vous devrez peut-être changer '0' par '1' si vous avez une caméra externe.
    cap = cv2.VideoCapture(0)

    # Vérifier si la caméra s'est bien ouverte
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam.")
        return

    print("\nLancement de l'analyse en temps réel.")
    print("Appuyez sur la touche 'q' pour quitter l'application.")

    # 3. Boucle infinie pour lire le flux vidéo image par image
    while True:
        # Lire une image de la webcam
        # 'ret' est un booléen (True si la capture a réussi), 'frame' est l'image elle-même
        ret, frame = cap.read()

        if not ret:
            print("Erreur : Impossible de lire le flux vidéo.")
            break

        # 4. Appeler l'IA pour analyser l'image en direct
        # stream=True optimise l'utilisation de la VRAM pour les flux vidéo.
        results = model(frame, device=0, stream=True)

        # 5. Dessiner les résultats directement sur l'image
        # Ultralytics fournit une méthode pratique : .plot()
        # Nous devons 'consommer' le générateur de résultats pour obtenir l'image annotée.
        for result in results:
            image_annotee = result.plot()

        # 6. Afficher l'image annotée dans une fenêtre Windows
        cv2.imshow("IA Vision - Flux Webcam Direct", image_annotee)

        # 7. Sortir de la boucle si l'utilisateur appuie sur la touche 'q'
        # '0xFF' garantit que nous lisons le bon code ASCII même sur des systèmes 64-bits
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 8. Nettoyage : Libérer la webcam et fermer toutes les fenêtres
    print("\nArrêt de l'application...")
    cap.release()
    cv2.destroyAllWindows()
    print("Terminé.")


if __name__ == "__main__":
    detecter_en_temps_reel()
