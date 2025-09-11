from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from .db import get_connection
import time

pedidos_routes = Blueprint("pedidos", __name__)



def get_user_id():
    return session.get("user_id")

# Ruta HTML para visualizar la página de pedidos
@pedidos_routes.route("/pedidos", methods=["GET"])
def mostrar_pedidos():
    return render_template("pedidos.html")


# Ruta API para obtener pedidos en formato JSON
@pedidos_routes.route("/pedidos", methods=["GET"])
def listar_pedidos():
    user_id = session.get("user_id")
    print(f"[DEBUG] user_id en sesión: {user_id}")
    
    if not user_id:
        return jsonify({"error": "No has iniciado sesión"}), 401

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pedidos WHERE usuario_id = %s ORDER BY fecha_pedido DESC", (user_id,))
        pedidos = cursor.fetchall()
        cursor.close()
        return jsonify(pedidos)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# -------------------------------
# GET /pedidos/<id> → Detalle de pedido
# -------------------------------
@pedidos_routes.route("/pedidos/<int:pedido_id>", methods=["GET"])
def detalle_pedido(pedido_id):
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "No has iniciado sesión"}), 401

    conn = get_connection()
    cursor = conn.cursor()

    # Verifica que el pedido sea del usuario
    cursor.execute("SELECT * FROM pedidos WHERE id = %s AND usuario_id = %s", (pedido_id, user_id))
    pedido = cursor.fetchone()
    if not pedido:
        cursor.close()
        return jsonify({"error": "Pedido no encontrado"}), 404

    # Obtener productos del pedido
    cursor.execute("""
        SELECT pp.producto_id, p.nombre, pp.cantidad, pp.precio_unitario
        FROM pedido_productos pp
        JOIN productos p ON pp.producto_id = p.id
        WHERE pp.pedido_id = %s
    """, (pedido_id,))
    productos = cursor.fetchall()

    cursor.close()

    return jsonify({
        "pedido": pedido,
        "productos": productos
    })


# -------------------------------
# POST /checkout → Procesar compra y registrar pedido
# -------------------------------
@pedidos_routes.route("/checkout", methods=["POST"])
def checkout():
    
    user_id = get_user_id()
    if not user_id:
        return jsonify({"error": "No has iniciado sesión"}), 401

    conn = get_connection()
    cursor = conn.cursor()

    # Obtener productos del carrito
    cursor.execute("""
        SELECT cp.producto_id, cp.cantidad, p.precio
        FROM carrito_productos cp
        JOIN carritos c ON cp.carrito_id = c.id
        JOIN productos p ON cp.producto_id = p.id
        WHERE c.usuario_id = %s
    """, (user_id,))
    items = cursor.fetchall()  # YA son dicts

    if not items:
        cursor.close()
        return jsonify({"error": "El carrito está vacío"}), 400

    total = sum(item["cantidad"] * float(item["precio"]) for item in items)

    time.sleep(2)  # Simular pago

    # Crear pedido
    cursor.execute("INSERT INTO pedidos (usuario_id, total) VALUES (%s, %s)", (user_id, total))
    pedido_id = cursor.lastrowid

    for item in items:
        cursor.execute("""
            INSERT INTO pedido_productos (pedido_id, producto_id, cantidad, precio_unitario)
            VALUES (%s, %s, %s, %s)
        """, (pedido_id, item["producto_id"], item["cantidad"], item["precio"]))

        cursor.execute("""
            UPDATE productos
            SET stock = stock - %s
            WHERE id = %s
        """, (item["cantidad"], item["producto_id"]))

    # Vaciar carrito
    cursor.execute("""
        DELETE cp FROM carrito_productos cp
        JOIN carritos c ON cp.carrito_id = c.id
        WHERE c.usuario_id = %s
    """, (user_id,))

    conn.commit()
    cursor.close()

    return jsonify({
        "mensaje": "Compra realizada correctamente",
        "pedido_id": pedido_id
    })

