import os
import numpy as np
from scipy.io import wavfile

# CORREÇÃO: Alinhado com o "GSC_v2" definido no script de download
def create_silence_class(dataset_dir="GSC_v2", segment_length_sec=1):
    noise_dir = os.path.join(dataset_dir, "_background_noise_")
    silence_out_dir = os.path.join(dataset_dir, "silence")
    
    if not os.path.exists(silence_out_dir):
        os.makedirs(silence_out_dir)

    # Listar ficheiros de ruído (ignorando o README)
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    
    print(f"A segmentar {len(noise_files)} ficheiros de fundo...")

    for noise_file in noise_files:
        sample_rate, data = wavfile.read(os.path.join(noise_dir, noise_file))
        
        # Calcular quantos clips de 1 segundo cabem no ficheiro
        samples_per_segment = int(segment_length_sec * sample_rate)
        num_segments = len(data) // samples_per_segment
        
        for i in range(num_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = data[start:end]
            
            # Guardar o novo clip na pasta 'silence'
            output_filename = f"silence_{noise_file[:-4]}_{i}.wav"
            wavfile.write(os.path.join(silence_out_dir, output_filename), sample_rate, segment)
            
    print(f"Classe 'silence' criada com sucesso em: {silence_out_dir}")

if __name__ == "__main__":
    # Verifica se a pasta do download existe antes de tentar segmentar
    if os.path.exists("GSC_v2"):
        create_silence_class()
    else:
        print("ERRO: A pasta GSC_v2 não foi encontrada. Garanta que correu o download primeiro.")