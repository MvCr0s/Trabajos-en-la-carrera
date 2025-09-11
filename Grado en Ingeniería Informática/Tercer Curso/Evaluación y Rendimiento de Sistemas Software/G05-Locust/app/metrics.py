from flask import Blueprint,  jsonify

metrics_bp = Blueprint("metrics", __name__)

# Variables simuladas
pedidos_creados = 15
usuarios_registrados = 4

@metrics_bp.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "pedidos_creados": pedidos_creados,
        "usuarios_registrados": usuarios_registrados
    })

