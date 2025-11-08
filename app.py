# ==========================================
# 🤖 API de Siver - Flask
# ==========================================

from flask import Flask, request, jsonify
import torch
import torch.nn as nn

# --- Definir red neuronal Siver (misma que entrenaste)
class SiverNet(nn.Module):
    def __init__(self):
        super(SiverNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),  # ejemplo simple (ajústalo a tu caso real)
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.model(x)

app = Flask(__name__)

# Cargar modelo entrenado
siver = SiverNet()
siver.load_state_dict(torch.load("siver_model.pth", map_location='cpu'))
siver.eval()

@app.route("/")
def home():
    return "🧠 Siver AI está activa."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Aquí adaptas según tu tipo de entrada
    x = torch.tensor([data["input"]], dtype=torch.float32)
    y = siver(x).item()
    return jsonify({"respuesta": y})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
