import torch

def compare_saved_batches():
    print("Cargando lotes para comparación...")
    data_v1 = torch.load('batch_v1.pt')
    data_v2 = torch.load('batch_v2.pt')

    in1, tar1 = data_v1['inputs'], data_v1['targets']
    in2, tar2 = data_v2['inputs'], data_v2['targets']

    # 1. Comprobar formas
    print(f"Shape Inputs -> V1: {in1.shape} | V2: {in2.shape}")
    print(f"Shape Targets -> V1: {tar1.shape} | V2: {tar2.shape}")

    # 2. Comprobación numérica estricta con torch.allclose
    inputs_close = torch.allclose(in1, in2, rtol=1e-5, atol=1e-6)
    targets_close = torch.allclose(tar1, tar2, rtol=1e-5, atol=1e-6)

    max_diff_in = (in1 - in2).abs().max().item()
    max_diff_tar = (tar1 - tar2).abs().max().item()

    print("\n--- Resultados de Paridad ---")
    print(f"¿Los Inputs (historia Y) son idénticos?: {inputs_close} (Dif. máx: {max_diff_in:.2e})")
    print(f"¿Los Targets (X, Z rápidos) son idénticos?: {targets_close} (Dif. máx: {max_diff_tar:.2e})")

if __name__ == '__main__':
    compare_saved_batches()