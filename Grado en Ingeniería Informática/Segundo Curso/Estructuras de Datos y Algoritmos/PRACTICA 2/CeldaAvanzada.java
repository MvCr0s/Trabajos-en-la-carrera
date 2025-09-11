/**
 * Hecho por Marcos de Diego y Alfredo del Val 
 * La clase CeldaAvanzada representa una cuadrícula de celdas conductoras.
 * Utiliza una variante del algoritmo de Disjoin-Sets para detectar cortocircuitos.
 */
public class CeldaAvanzada implements Celda {

    // Variables de instancia
    private int n = -1;
    private int m = -1;
    private boolean[][] conductor;
    private int[] vencino;
    private int[] rank;
    private boolean roto;
    private int ui, uj;

    // Vecinos predefinidos
    private static final int[][] VECINOS = {
        {-1, 0}, {1, 0}, {1, 1}, {1, -1},
        {0, -1}, {0, 1}, {-1, 1}, {-1, -1}
    };

    /**
     * Inicializa la cuadrícula con un tamaño específico n*m.
     * Crea las estructuras de datos necesarias y establece una conexión entre la primera y última fila.
     * @param n Número de filas
     * @param m Número de columnas
     */
    @Override
    public void Inicializar(int n, int m) {
        if (this.n != n || this.m != m) {
            this.n = n;
            this.m = m;
            conductor = new boolean[n][m];
            vencino = new int[n * m];
            rank = new int[n * m];
            Inicialización();
            conectaPrimerosYUltimos();
        } else {
            reiniciar();
        }
        roto = false;
        ui = -1;
        uj = -1;
    }

    /**
     * Marca la celda en la posición (i, j) como conductora y establece conexiones con sus vecinos.
     * Verifica si hay un camino conductor desde la primera hasta la última fila.
     * @param i Índice de fila
     * @param j Índice de columna
     */
    @Override
    public void RayoCosmico(int i, int j) {
        if (i < 0 || i >= n || j < 0 || j >= m || conductor[i][j]) {
            // Verificar si las coordenadas están fuera de los límites o si la celda ya es conductora
            return;
        }

        ui = i;
        uj = j;
        conductor[i][j] = true;

        boolean hayConexiones = conectaConVecinos(i, j);
        roto = roto || (hayConexiones && HayCamino());
    }

    /**
     * Devuelve si hay un cortocircuito en la cuadrícula.
     * @return true si hay cortocircuito, false de lo contrario
     */
    @Override
    public boolean Cortocircuito() {
        return roto;
    }

    /**
     * Calcula el índice en el array 1D correspondiente a la posición (i, j) en la cuadrícula 2D.
     * @param i Índice de fila
     * @param j Índice de columna
     * @return Índice en el array 1D
     */
    protected int index(int i, int j) {
        return i * m + j;
    }

    /**
     * Inicializa los arrays de padres y rangos para el algoritmo de Union-Find.
     */
    private void Inicialización() {
        for (int i = 0; i < n * m; i++) {
            vencino[i] = i;
            rank[i] = 0;
        }
    }

    /**
     * Conecta la celda en la posición (i, j) con sus vecinos conductores y devuelve si hubo conexiones.
     * @param i Índice de fila
     * @param j Índice de columna
     * @return true si hubo conexiones con vecinos conductores, false de lo contrario
     */
    private boolean conectaConVecinos(int i, int j) {
        boolean hayConexiones = false;

        for (int[] desp : VECINOS) {
            int ni = i + desp[0];
            int nj = j + desp[1];
            if (cedaValida(ni, nj) && conductor[ni][nj]) {
                hayConexiones |= unir(index(i, j), index(ni, nj));
            }
        }
        return hayConexiones;
    }

    /**
     * Une los conjuntos de elementos que contienen a x y y.
     * @param x Elemento x
     * @param y Elemento y
     * @return true si los conjuntos de x e y no eran iguales y se unieron, false de lo contrario
     */
    private boolean unir(int x, int y) {
        int raizX = encontrar(x);
        int raizY = encontrar(y);

        if (raizX != raizY) {
            if (rank[raizX] < rank[raizY]) {
                vencino[raizX] = raizY;
            } else if (rank[raizX] > rank[raizY]) {
                vencino[raizY] = raizX;
            } else {
                vencino[raizX] = raizY;
                rank[raizY]++;
            }
            return true;
        }
        return false;
    }

    /**
     * Encuentra el representante del conjunto al que pertenece x utilizando compresión de ruta.
     * @param x Elemento x
     * @return Representante del conjunto al que pertenece x
     */
    private int encontrar(int x) {
        if (vencino[x] != x) {
            vencino[x] = encontrar(vencino[x]);
        }
        return vencino[x];
    }

    /**
     * Verifica si hay un camino conductor desde la primera hasta la última fila.
     * @return true si hay un camino conductor, false de lo contrario
     */
    private boolean HayCamino() {
        int raiz0 = encontrar(0);
        int raizN = encontrar(n * (m - 1));

        return raiz0 == raizN;
    }

    /**
     * Verifica si la celda en la posición (i, j) es válida en la cuadrícula.
     * @param i Índice de fila
     * @param j Índice de columna
     * @return true si la celda es válida, false de lo contrario
     */
    private boolean cedaValida(int i, int j) {
        return i >= 0 && i < n && j >= 0 && j < m;
    }

    /**
     * Conecta la primera y última fila de la cuadrícula.
     */
    private void conectaPrimerosYUltimos() {
        for (int j = 0; j < m; j++) {
            unir(0, j);
            unir((n - 1) * m, (n - 1) * m + j);
        }
    }

    /**
     * Reinicia la cuadrícula, marcando todas las celdas como no conductoras
     * y restableciendo las conexiones entre la primera y última fila.
     */
    public void reiniciar() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                conductor[i][j] = false;
            }
        }
        Inicialización(); // También reinicia las estructuras de datos internas
        conectaPrimerosYUltimos(); // Vuelve a conectar la primera y última fila
        roto = false;
        ui = -1;
        uj = -1;
    }

    /**
     * Representación visual de la cuadrícula.
     * Marca la celda actual del rayo cósmico con "*", las celdas conductoras con "X",
     * y las no conductoras con "·".
     * @return Representación visual de la cuadrícula
     */
    @Override
    public String toString() {
        char[][] gridRepresentation = new char[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                gridRepresentation[i][j] = (i == ui && j == uj) ? '*' : (conductor[i][j] ? 'X' : '·');
            }
        }

        StringBuilder res = new StringBuilder();
        for (char[] row : gridRepresentation) {
            res.append(row).append("\n");
        }
        return res.toString();
    }
}
