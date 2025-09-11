from flask import Flask, session
from app.inicio import inicio_routes
from app.registro import bp
from app.productos import productos_routes
from app.carrito import carrito_routes
from app.pedidos import pedidos_routes
from app.menu import menu_routes
import os




def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    # Importar y registrar Blueprints 
    from .routes import inicio, registro,productos, carrito, pedidos

    app.register_blueprint(inicio_routes)
    app.register_blueprint(bp)
    app.register_blueprint(menu_routes)
    app.register_blueprint(productos_routes)
    app.register_blueprint(carrito_routes)
    app.register_blueprint(pedidos_routes)
    
    return app

