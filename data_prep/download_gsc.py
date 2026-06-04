import os
import requests
import tarfile

# URL real e direto para o arquivo compactado de 2.4GB
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ficheiro_tar = "gsc_v2.tar.gz"
pasta_destino = "GSC_v2"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

print("A iniciar o download real do dataset (aprox. 2.4GB)...")
print("Por favor, aguarde. O processo pode demorar dependendo da sua internet.")

try:
    # Remove qualquer arquivo corrompido residual antes de reiniciar
    if os.path.exists(ficheiro_tar) and os.path.getsize(ficheiro_tar) < 1000000:
        os.remove(ficheiro_tar)
        
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(ficheiro_tar, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                print(".", end="", flush=True)
                
    print("\nDownload concluído com sucesso!")
    print("A extrair ficheiros na pasta GSC_v2... (Aguarde alguns minutos)")
    
    with tarfile.open(ficheiro_tar, "r:gz") as tar:
        tar.extractall(path=pasta_destino)
        
    print(f"\nSucesso absoluto! Pasta pronta em: {os.path.abspath(pasta_destino)}")

except Exception as e:
    print(f"\nErro durante o processo: {e}")
    print(f"Pode descarregar manualmente colando este link no seu navegador: {url}")