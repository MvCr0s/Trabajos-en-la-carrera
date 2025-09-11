from flask import Blueprint, jsonify, request, render_template, session, redirect, url_for
from .db import get_connection

productos_routes = Blueprint("productos", __name__)

# HTML de productos
@productos_routes.route("/productos", methods=["GET"])
def mostrar_productos():
    return render_template("productos.html")

# API JSON de productos
@productos_routes.route("/productos", methods=["GET"])
def listar_productos():
    try:
        categoria = request.args.get("categoria")
        conn = get_connection()
        cursor = conn.cursor()

        if categoria:
            cursor.execute("SELECT * FROM productos WHERE stock > 0 AND categoria = %s", (categoria,))
        else:
            cursor.execute("SELECT * FROM productos WHERE stock > 0")

        productos = cursor.fetchall()
        cursor.close()
        return jsonify(productos)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    



# -------------------------------
# GET /productos/<id> → Detalle de producto
# -------------------------------
@productos_routes.route("/productos/<int:producto_id>", methods=["GET"])
def detalle_producto(producto_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM productos WHERE id = %s", (producto_id,))
        rows = cursor.fetchall()

        cursor.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# PUT /productos/<id> → Actualizar producto completamente
# -------------------------------
@productos_routes.route("/productos/<int:producto_id>", methods=["PUT"])
def actualizar_producto(producto_id):
    if "user_id" not in session:
        return jsonify({"error": "No autorizado"}), 401

    data = request.get_json()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE productos
        SET nombre = %s, precio = %s, stock = %s, descripcion = %s
        WHERE id = %s
    """, (data.get("nombre"), data.get("precio"), data.get("stock"), data.get("descripcion"), producto_id))

    conn.commit()
    cursor.close()
    return jsonify({"mensaje": "Producto actualizado correctamente"}), 200

# -------------------------------
# PATCH /productos/<id> → Actualizar producto parcialmente
# -------------------------------
@productos_routes.route("/productos/<int:producto_id>", methods=["PATCH"])
def modificar_producto(producto_id):
    if "user_id" not in session:
        return jsonify({"error": "No autorizado"}), 401

    data = request.get_json()
    conn = get_connection()
    cursor = conn.cursor()

    updates = []
    params = []
    for key, value in data.items():
        if key in ["nombre", "precio", "stock", "descripcion"]:
            updates.append(f"{key} = %s")
            params.append(value)

    if not updates:
        return jsonify({"error": "No se proporcionaron campos válidos para actualizar"}), 400

    params.append(producto_id)
    query = f"UPDATE productos SET {', '.join(updates)} WHERE id = %s"
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
    return jsonify({"mensaje": "Producto modificado correctamente"}), 200

# -------------------------------
# GET /editarProducto → Mostrar formulario de edición
# -------------------------------
@productos_routes.route("/editarProducto")
def mostrar_formulario_edicion():
    if "user_id" not in session:
        return redirect(url_for("inicio.mostrar_inicio"))
    return render_template("editarProducto.html")




@productos_routes.route("/menu")
def vista_menu():
    if "user_id" not in session:
        return redirect(url_for("inicio.mostrar_inicio"))

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM productos WHERE stock > 0")
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

    productos = [dict(zip(column_names, row)) for row in rows]  #   QUITA EL `if row[0] != "id"`

    cursor.close()

    return render_template("menu.html", productos=productos)
