from flask import Blueprint, render_template, request, jsonify, session
from app.db import get_connection
import hashlib

bp = Blueprint('registro', __name__)

@bp.route('/registro', methods=['POST'])
def registro():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    email = data.get("email", f"{username}@fake.com")

    if not username or not password:
        return jsonify({"error": "Faltan datos"}), 400



    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO usuarios (nombre, email, contraseña_hash) VALUES (%s, %s, %s)",
            (username, email, password)
        )
        conn.commit()
        session["user_id"] = cursor.lastrowid
        session["username"] = username
        return jsonify({"mensaje": "Usuario registrado"}), 201
    except Exception as e:
        if "Duplicate" in str(e):
            return jsonify({"error": "Usuario ya existe"}), 409
        return jsonify({"error": "Error interno"}), 500
    finally:
        cursor.close()
        conn.close()

@bp.route('/registro', methods=['GET'])
def mostrar_registro():
    return render_template('registro.html')
