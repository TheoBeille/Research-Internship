import torch
import gc
from Algo_setuptorch import Params
from data.dataset_mayo import build_train_test_data_mayo
from algorithm.unrolled_model import UnrolledFBS

def diagnose_memory():
    # Configuration du périphérique
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DIAGNOSTIC MÉMOIRE VRAM ===")
    print(f"Périphérique utilisé : {device}")

    if device.type != "cuda":
        print("Erreur : Ce script de diagnostic nécessite un GPU CUDA.")
        return

    # Nettoyage initial du cache CUDA
    torch.cuda.empty_cache()
    gc.collect()

    print("\n[1] État initial de la mémoire :")
    print(f"  - Allouée : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
    print(f"  - Réservée : {torch.cuda.memory_reserved(device) / (1024**2):.2f} MB")

    # Paramètres et géométrie (identiques à ton main.py)
    params = Params(size=512)
    N_ANGLES = 180
    size = params.size
    SHAPES = [
        (1, 1, size, size),   # u (image)
        (1, 2, size, size),   # w
        (1, 2, size, size),   # p
        (1, 3, size, size),   # q
    ]
    N_CH_primal = sum(s[1] for s in SHAPES[:2])

    print("\n[2] Chargement d'un échantillon de test (Mayo dataset)...")
    _, test_data = build_train_test_data_mayo(
        train_patients=["L004"],
        test_patients=["L014"],
        params=params,
        device=device,
        n_angles=N_ANGLES,
        cache_dir="./data/mayo_cache_512",
        noise_level=0.05,
        max_train_slices=1,
        max_test_slices=1,
    )

    initial_state, clean, functions = test_data[0]
    initial_state = initial_state.to(device)

    print("\n[3] Instanciation du modèle déroulé (UnrolledFBS avec T=10)...")
    model = UnrolledFBS(
        params=params,
        shapes=SHAPES,
        n_channels=N_CH_primal,
        T=10,
        alpha=0.99,
    ).to(device).float()

    print(f"  - Mémoire après instanciation du modèle : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")

    # Réinitialiser les statistiques de pic pour capturer l'exécution
    torch.cuda.reset_peak_memory_stats(device)

    print("\n[4] Exécution de la passe avant (Forward Pass)...")
    try:
        with torch.set_grad_enabled(True):
            AxCx, residuals, objectives, x_final = model(initial_state, functions)
            loss = AxCx[-1]
        
        print(f"  - Mémoire active après Forward : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        print(f"  - Pic VRAM atteint pendant le Forward : {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")

        print("\n[5] Exécution de la passe arrière (Backward Pass / loss.backward())...")
        loss.backward()
        
        print(f"  - Mémoire active après Backward : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        print(f"  - Pic VRAM global (Forward + Backward) : {torch.cuda.max_memory_allocated(device) / (1024**2):.2f} MB")
        print("\n[SUCCÈS] Le test de mémoire s'est déroulé sans erreur OOM.")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n[ÉCHEC] CUDA Out of Memory (OOM) détecté !")
            print("-" * 50)
            # Afficher le résumé détaillé de la mémoire PyTorch au moment du crash
            print(torch.cuda.memory_summary(device=device, abbreviated=False))
            print("-" * 50)
        else:
            raise e

if __name__ == "__main__":
    diagnose_memory()