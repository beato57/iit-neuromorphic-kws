import os
import sys
import numpy as np
import tensorflow as tf
import nengo
import nengo_dl
from sklearn.model_selection import train_test_split

# 1. Configuração de Caminhos e Carregamento de Dados Reais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.load_features import carregar_e_extrair_caracteristicas

print("A carregar características MFCC extraídas do dataset GSC_v2...")

caminho_gsc = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GSC_v2'))
X_raw, y_raw = carregar_e_extrair_caracteristicas(caminho_gsc)

if len(X_raw) == 0:
    print("ERRO: Nenhuma amostra válida foi carregada.")
    sys.exit(1)

# Mapeamento estrito das 12 classes textuais para IDs numéricos (0 a 11)
labels_ordem = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes", "silence", "unknown"]
label_to_id = {name: i for i, name in enumerate(labels_ordem)}
y_numeric = np.array([label_to_id[lbl] for lbl in y_raw])

# Ajuste dimensional para o Nengo-DL: (amostras, timesteps, 1)
y_data_t = np.repeat(y_numeric[:, np.newaxis, np.newaxis], 100, axis=1)
X_data = X_raw

# Divisão do dataset em Treino (80%) e Teste/Validação (20%)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_t, test_size=0.2, random_state=42)

# 2. Definição da Arquitetura SNN Neuromórfica no Nengo
n_neurons = 200      
n_features = 13      
n_classes = 12       

model = nengo.Network(label="SNN_KWS_Reentrant_Official")

with model:
    input_node = nengo.Node(output=np.zeros(n_features), label="input_node")
    
    lif_neurons = nengo.LIF(tau_rc=0.02, tau_ref=0.002, amplitude=0.01)
    ens = nengo.Ensemble(n_neurons, dimensions=n_features, neuron_type=lif_neurons, label="middle_layer")
    
    nengo.Connection(input_node, ens, synapse=None)
    nengo.Connection(ens, ens, synapse=0.02, transform=0.4) # CONEXÃO REENTRANTE (Φ)
    
    output_node = nengo.Node(size_in=n_classes, label="output_node")
    nengo.Connection(ens, output_node, synapse=0.01, transform=np.zeros((n_classes, n_features)))
    
    output_probe = nengo.Probe(output_node, label="output_probe")

# 3. Definição de Função de Perda Customizada Temporal para Nengo-DL
# Esta função força o Keras a olhar apenas para o último passo de tempo, ignorando ruídos intermédios
def loss_ultimo_timestep(y_true, y_pred):
    # y_true tem formato (batch, timesteps, 1) -> extrai último timestep
    y_true_final = y_true[:, -1, 0]
    # y_pred tem formato (batch, timesteps, classes) -> extrai último timestep
    y_pred_final = y_pred[:, -1, :]
    
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true_final, y_pred_final, from_logits=True
    )

def accuracy_ultimo_timestep(y_true, y_pred):
    y_true_final = y_true[:, -1, 0]
    y_pred_final = y_pred[:, -1, :]
    return tf.keras.metrics.sparse_categorical_accuracy(y_true_final, y_pred_final)

# 4. Otimização e Treino via Nengo-DL
print("A inicializar o simulador Nengo-DL...")
with nengo_dl.Simulator(model, minibatch_size=32) as sim:
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compilação vinculando as funções customizadas temporais explicitamente ao probe
    sim.compile(
        optimizer=optimizer, 
        loss={output_probe: loss_ultimo_timestep}, 
        metrics={output_probe: accuracy_ultimo_timestep}
    )
    
    print("A iniciar o treino oficial da SNN com os dados reais...")
    sim.fit(x={input_node: X_train}, y={output_probe: y_train}, epochs=10)
    
    print("\nA avaliar o modelo no conjunto de validação...")
    eval_results = sim.evaluate(x={input_node: X_test}, y={output_probe: y_test})
    
    print(f"\n--- RESULTADOS DO PAPER ---")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Salvaguarda automática dos parâmetros sinápticos
    os.makedirs("parameters", exist_ok=True)
    sim.save_params("parameters/snn_kws_weights")
    print("Sucesso! Pesos sinápticos guardados em 'models/parameters/snn_kws_weights'.")
    
    # Guardar os dados de teste reais para o script de resultados
    os.makedirs("../results/test_data", exist_ok=True)
    np.save("../results/test_data/X_test.npy", X_test)
    np.save("../results/test_data/y_test.npy", y_test)
    print("Dados de validação exportados para a pasta 'results/test_data/'.")