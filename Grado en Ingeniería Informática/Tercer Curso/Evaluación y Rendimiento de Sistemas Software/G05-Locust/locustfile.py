from locust import HttpUser, task, between
import random
import uuid
import string
import hashlib
import json

def generar_usuario_aleatorio():
    """
    Genera un nombre de usuario aleatorio y una contraseña por defecto.

    Returns:
        tuple: (username, password) generados automáticamente.
    """
    
    nombre = "user" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    contraseña = "test1234"
    return nombre, contraseña

class UsuarioSimulado(HttpUser):
    
    """
    Simula el comportamiento típico de un usuario real en el supermercado.

    El usuario se registra, inicia sesión, visualiza productos, agrega al carrito
    y finaliza una compra. Se utiliza para simular una carga de tráfico realista.
    """
    
    wait_time = between(1, 3)

    def on_start(self):
        """
        Método que se ejecuta al comenzar la simulacion.
        Registra un usuario nuevo y realiza login automatico.
        """
        self.username, self.password = generar_usuario_aleatorio()
        self.registrar_usuario()
        self.login()

    def registrar_usuario(self):
        """
        Registra un nuevo usuario en la API simulando el alta de un cliente.
        """
        
        email = f"{self.username}@test.com"
        payload = {
            "username": self.username,
            "password": self.password,
            "email": email
        }
        headers = {'Content-Type': 'application/json'}

        with self.client.post("/registro", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 201:
                response.success()
                print(f"[OK] Registrado: {self.username}")
            else:
                response.failure(f"Fallo en registro: {response.status_code}")

    def login(self):
        
        """
        Inicia sesión con el usuario recién registrado.
        """
        payload = {
            "username": self.username,
            "password": self.password  
        }
        headers = {'Content-Type': 'application/json'}

        with self.client.post("/inicio", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"[OK] Sesión iniciada con {self.username}")
            else:
                response.failure("Login fallido")
                print(f"[ERROR] Falló login con {self.username}")



    def obtener_producto_id(self):
        
        """
        Devuelve un ID de producto aleatorio dentro del rango disponible.

        Returns:
            int: ID del producto.
        """
        
        return random.randint(13, 50)

    @task(2)
    def ver_menu(self):
        """Simula una petición GET al menú de la aplicación."""
               
        self.client.get("/menu", name="GET /menu")

    @task(3)
    def ver_productos(self):
        """Simula la visualización del catálogo de productos."""   
             
        self.client.get("/productos", name="GET /productos")

    @task(2)
    def ver_carrito(self):
        """Consulta el estado actual del carrito de compras."""
        
        self.client.get("/carrito", name="GET /carrito")

    @task(1)
    def agregar_carrito(self):
        
        """Agrega un producto al carrito con una cantidad por defecto de 1."""
          
        producto_id = self.obtener_producto_id()
        self.client.post("/carrito/agregar", json={
            "producto_id": producto_id,
            "cantidad": 1
        }, name=f"POST /carrito/agregar")

    @task(1)
    def procesar_compra(self):
        
        """
        Simula la acción de agregar un producto al carrito
        y luego completar la compra con un checkout.
        """
               
        self.agregar_carrito()
        self.client.post("/checkout", json={}, name="POST /checkout")


    @task(1)
    def ver_pedidos(self):
        """Consulta todos los pedidos del usuario autenticado."""
        
        self.client.get("/pedidos", name="GET /pedidos")

class UsuarioAPI(HttpUser):
    
    """
    Simula a un administrador o usuario avanzado que gestiona productos.

    Incluye tareas adicionales como actualización parcial o total de productos.
    """
    
    wait_time = between(1, 3)

    def on_start(self):
        """
        Se ejecuta al inicio de la simulación.
        Registra un nuevo usuario y realiza login.
        """
        
        self.username = f"user_{uuid.uuid4().hex[:8]}"
        self.email = f"{self.username}@example.com"
        self.password = "123"
        self.ultimo_pedido_id = None

        self.registro()
        self.inicio()

    def registro(self):
        """Registra el usuario actual en el sistema."""
        payload = {
            "username": self.username,
            "email": self.email,
            "password": self.password
        }
        with self.client.post("/registro", json=payload, name="POST /registro", catch_response=True) as response:
            if response.status_code != 201:
                response.failure(f"Fallo en registro: {response.status_code}")
            else:
                response.success()

    def inicio(self):
        """Realiza login con el usuario creado."""
        payload = {"username": self.username, "password": self.password}  # ✔️ texto plano


        with self.client.post("/inicio", json=payload, name="POST /inicio", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("Fallo en inicio")
            else:
                self.client.get("/menu")

    def obtener_producto_id(self):
        
        """
        Retorna un ID aleatorio de producto para pruebas de edición.

        Returns:
            int: ID del producto.
        """
        return random.randint(6, 10)

    @task(2)
    def put_producto(self):
        """Simula una actualización completa de un producto vía PUT."""
        producto_id = self.obtener_producto_id()
        self.client.put(f"/productos/{producto_id}", json={
            "nombre": "Producto Actualizado",
            "precio": round(random.uniform(10.0, 100.0), 2),
            "stock": random.randint(1, 100),
            "descripcion": "Actualizado por test"
        }, name=f"PUT /productos/{producto_id}")

    @task(2)
    def patch_producto(self):
        """Simula una modificación parcial del stock de un producto vía PATCH."""
        
        producto_id = self.obtener_producto_id()
        self.client.patch(f"/productos/{producto_id}", json={
            "stock": random.randint(1, 100),
        }, name=f"PATCH /productos/{producto_id}")

    @task(1)
    def vaciar_carrito(self):
        """Simula el vaciado completo del carrito de compras."""

        self.client.delete("/carrito/vaciar", name="DELETE /carrito/vaciar")

    @task(1)
    def ver_pedido_detalle(self):
        """
        Consulta los detalles del último pedido realizado,
        si es que existe.
        """        
        
        if self.ultimo_pedido_id:
            self.client.get(f"/pedidos/{self.ultimo_pedido_id}", name="GET /pedidos/:id")

    @task(1)
    def checkout(self):
        """
        Simula el proceso de checkout para confirmar un pedido.

        Guarda el ID del pedido si se realiza correctamente.
        """
        
        with self.client.post("/checkout", name="POST /checkout", catch_response=True) as response:
            if response.status_code == 200:
                self.ultimo_pedido_id = response.json().get("pedido_id")
                response.success()
            else:
                response.failure("Fallo en checkout")
