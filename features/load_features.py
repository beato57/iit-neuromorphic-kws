import os
import librosa
import numpy as np

def carregar_e_extrair_caracteristicas(diretorio_base="GSC_v2"):
    dados_mfcc = []
    etiquetas = []
    
    # 1. Definir estritamente os 10 comandos alvo do paper
    comandos_alvo = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    
    # Listar todas as pastas dentro do dataset (excluindo pastas ocultas)
    todas_pastas = [d for d in os.listdir(diretorio_base) 
                    if os.path.isdir(os.path.join(diretorio_base, d)) and not d.startswith('_')]
    
    print("A iniciar o carregamento e extração de MFCCs...")
    
    for pasta in todas_pastas:
        caminho_classe = os.path.join(diretorio_base, pasta)
        print(f"A processar diretório: {pasta} ...")
        
        # Determinar a etiqueta correta com base no mapeamento do paper (12 classes)
        if pasta in comandos_alvo:
            label_final = pasta
        elif pasta == "silence":
            label_final = "silence"
        else:
            label_final = "unknown" # Agrupa as restantes 24 palavras auxiliares
            
        for ficheiro in os.listdir(caminho_classe):
            if ficheiro.endswith('.wav'):
                caminho_wav = os.path.join(caminho_classe, ficheiro)
                
                try:
                    # Carrega o áudio a 16kHz (nativo do dataset)
                    audio, sr = librosa.load(caminho_wav, sr=16000)
                    
                    # Garante a consistência temporal de exatamente 1 segundo (16000 amostras)
                    if len(audio) == 16000:
                        # Extração de MFCCs (Janela 30ms, Passo 10ms -> 100 frames)
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=480, hop_length=160)
                        
                        # Normalização Z-score para estabilidade dos Spikes no Nengo
                        mfcc_norm = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                        
                        # Guardar a transposta para o formato temporal do Nengo: (frames, features)
                        dados_mfcc.append(mfcc_norm.T)
                        etiquetas.append(label_final)
                except Exception as e:
                    # Ignorar ficheiros eventualmente corrompidos
                    continue
                    
    return np.array(dados_mfcc), np.array(etiquetas)

if __name__ == "__main__":
    if os.path.exists("GSC_v2"):
        X, y = carregar_e_extrair_caracteristicas()
        print(f"\nSucesso! Extraídos MFCCs de {X.shape[0]} amostras estáveis.")
        print(f"Formato da Matriz de Entrada X: {X.shape} -> (Amostras, Frames, MFCC_Coefs)")
        print(f"Distribuição única de Etiquetas: {np.unique(y)}")
    else:
        print("ERRO: Execute primeiro os scripts da pasta data_prep.")