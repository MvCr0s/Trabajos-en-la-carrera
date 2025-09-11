from flask import Blueprint, render_template, request, jsonify, session
import hashlib
from app.db import get_connection

inicio_routes = Blueprint("inicio", __name__)

@inicio_routes.route("/inicio", methods=["GET"])
def mostrar_inicio():
    return render_template("inicio.html")


@inicio_routes.route('/inicio', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM usuarios WHERE nombre=%s AND contraseña_hash=%s",
                   (username, password))
    user = cursor.fetchone()
    if user:
        session["user_id"] = user["id"]
        session["username"] = username
        return jsonify({"mensaje": "Login exitoso"}), 200
    return jsonify({"error": "Credenciales inválidas"}), 401
