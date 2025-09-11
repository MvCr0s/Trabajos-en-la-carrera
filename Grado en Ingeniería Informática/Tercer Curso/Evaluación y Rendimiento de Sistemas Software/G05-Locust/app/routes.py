from flask import Blueprint, render_template, redirect, url_for

main = Blueprint('main', __name__)

@main.route('/')
def inicio():
    return render_template('inicio.html')

@main.route('/menu')
def menu():
    return render_template('menu.html')

@main.route('/registro')
def registro():
    return render_template('registro.html')

@main.route('/productos')
def productos():
    return render_template('productos.html')

@main.route('/carrito')
def carrito():
    return render_template('carrito.html')

@main.route('/pedidos')
def pedidos():
    return render_template('pedidos.html')