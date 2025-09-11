from flask import Blueprint, render_template, jsonify, session

menu_routes = Blueprint("menu", __name__)

@menu_routes.route("/menu")
def mostrar_menu():
    username = session.get("username", "Invitado")
    return render_template("menu.html", username=username)

@menu_routes.route("/menu")
def api_menu():
    # Datos simulados. Puedes conectarlo con BD si tienes tabla de categorías
    return jsonify({
        "descripcion": "¡Bienvenido a la tienda online de comida más sabrosa!",
        "categorias": ["Pizza", "Hamburguesas", "Ensaladas", "Postres", "Bebidas"]
    })



@menu_routes.route("/menu")
def vista_menu():
    if "user_id" not in session:
        return redirect(url_for("app.mostrar_inicio"))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM productos WHERE stock > 0")
    productos = cursor.fetchall()
    cursor.close()

    return render_template("menu.html", productos=productos)




