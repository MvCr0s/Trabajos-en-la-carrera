from flask import Blueprint, render_template, jsonify, request, session
from .db import get_connection
from pymysql.cursors import DictCursor

carrito_routes = Blueprint("carrito", __name__)

def get_user_id():
    return session.get("user_id")

@carrito_routes.route("/carrito", methods=["GET"])
def mostrar_carrito():
    return render_template("carrito.html")


# Crear o recuperar carrito del usuario
def obtener_o_crear_carrito(usuario_id):
    conn = get_connection() 
    cursor = conn.cursor(DictCursor)
    cursor.execute("SELECT id FROM carritos WHERE usuario_id = %s", (usuario_id,))
    row = cursor.fetchone()

    if row is None:
        cursor.execute("INSERT INTO carritos (usuario_id) VALUES (%s)", (usuario_id,))
        conn.commit()
        carrito_id = cursor.lastrowid
        print(f"Carrito nuevo creado con ID: {carrito_id}")
    else:
        carrito_id = row["id"]
        print(f"Carrito encontrado con ID: {carrito_id}")


    cursor.close()
    conn.close()
    return carrito_id




# POST /carrito/agregar
@carrito_routes.route("/carrito/agregar", methods=["POST"])
def agregar_al_carrito():
    try:
        user_id = get_user_id()
        if not user_id:
            return jsonify({"error": "No has iniciado sesión"}), 401

        data = request.get_json()
        producto_id = data.get("producto_id")
        cantidad = data.get("cantidad", 1)

        if not producto_id:
            return jsonify({"error": "Falta producto_id"}), 400

        carrito_id = obtener_o_crear_carrito(user_id)

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM carrito_productos
            WHERE carrito_id = %s AND producto_id = %s
        """, (carrito_id, producto_id))
        row = cursor.fetchone()

        if row:
            cursor.execute("""
                UPDATE carrito_productos
                SET cantidad = cantidad + %s
                WHERE carrito_id = %s AND producto_id = %s
            """, (cantidad, carrito_id, producto_id))
        else:
            cursor.execute("""
                INSERT INTO carrito_productos (carrito_id, producto_id, cantidad)
                VALUES (%s, %s, %s)
            """, (carrito_id, producto_id, cantidad))

        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"mensaje": "Producto agregado al carrito"}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# GET /carrito
@carrito_routes.route("/carrito", methods=["GET"])
def obtener_carrito():
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "No has iniciado sesión"}), 401

    carrito_id = obtener_o_crear_carrito(user_id)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.nombre, p.precio, cp.cantidad
        FROM carrito_productos cp
        JOIN productos p ON cp.producto_id = p.id
        WHERE cp.carrito_id = %s
    """, (carrito_id,))
    
    productos = cursor.fetchall()  # devuelve una lista de tuplas o dicts, depende del cursor
    cursor.close()

    return jsonify(productos)







# DELETE /carrito/vaciar
@carrito_routes.route("/carrito/vaciar", methods=["DELETE"])
def vaciar_carrito():
    try:
        user_id = get_user_id()
        if not user_id:
            return jsonify({"error": "No has iniciado sesión"}), 401

        carrito_id = obtener_o_crear_carrito(user_id)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM carrito_productos WHERE carrito_id = %s", (carrito_id,))
        conn.commit()
        cursor.close()
        return jsonify({"mensaje": "Carrito vaciado"}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno"}), 500
