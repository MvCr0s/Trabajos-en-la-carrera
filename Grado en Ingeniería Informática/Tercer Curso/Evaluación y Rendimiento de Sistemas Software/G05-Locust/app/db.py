import pymysql
from flask import g
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    return pymysql.connect(
            host="localhost",
            user="admin2",
            password="123412341234.",
            database="tienda_comida",
            cursorclass=pymysql.cursors.DictCursor  # para obtener resultados como diccionario
        )
    

def close_connection(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

