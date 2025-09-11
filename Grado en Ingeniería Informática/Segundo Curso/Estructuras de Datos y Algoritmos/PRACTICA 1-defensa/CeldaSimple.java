public class CeldaSimple implements Celda {

    private static final int[][] VECINOS = {
            {-1, -1},    {-1, 0},   {-1, 1},
            {0, -1},                {0, 1}, 
            {1, -1},     {1, 0},    {1, 1} 
    };

    private int n = -1;
    private boolean[][] conductor;
    private int[] parent;
    private int[] rank;
    private boolean roto;
    private int ui, uj;

    @Override
    public void Inicializar(int n) {
        if (this.n != n) {
            this.n = n;
            conductor = new boolean[n][n];
            parent = new int[n * n];
            rank = new int[n * n];
            initializeArrays();
            connectFirstAndLastRow();
        }
        roto = false;
        ui = -1;
        uj = -1;
    }

    @Override
    public void RayoCosmico(int i, int j) {
        if (conductor[i][j]) return;
        ui = i;
        uj = j;
        conductor[i][j] = true;

        boolean hayConexiones = connectWithNeighbors(i, j);
        roto = roto || (hayConexiones && HayCamino());

        if(roto) reiniciar();
    }

    @Override
    public boolean Cortocircuito() {
        return roto;
    }

    protected int index(int i, int j) {
        return i * n + j;
    }

    private void initializeArrays() {
        for (int i = 0; i < n * n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    private boolean connectWithNeighbors(int i, int j) {
        boolean hayConexiones = false;
        for (int[] desp : VECINOS) {
            int ni = i + desp[0];
            int nj = j + desp[1];
            if (isValidCell(ni, nj) && conductor[ni][nj]) {
                hayConexiones |= unir(index(i, j), index(ni, nj));
            }
        }
        return hayConexiones;
    }

    private boolean unir(int x, int y) {
        int rootX = encontrar(x);
        int rootY = encontrar(y);
        if (rootX != rootY) {
            if (rank[rootX] < rank[rootY]) parent[rootX] = rootY;
            else if (rank[rootX] > rank[rootY]) parent[rootY] = rootX;
            else {
                parent[rootX] = rootY;
                rank[rootY]++;
            }
            return true;
        }
        return false;
    }

    private int encontrar(int x) {
        return parent[x] != x ? (parent[x] = encontrar(parent[x])) : x;
    }

    private boolean HayCamino() {
        int root0 = encontrar(0);
        for (int k = 0; k < n; k++) {
            if (encontrar(index(n - 1, k)) == root0) return true;
        }
        return false;
    }

    private boolean isValidCell(int i, int j) {
        return i >= 0 && i < n && j >= 0 && j < n;
    }

    private void connectFirstAndLastRow() {
        for (int j = 0; j < n; j++) {
            unir(0, j);
            unir(n - 1, j);
        }
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res.append(i == ui && j == uj ? "*" : conductor[i][j] ? "X" : "·");
            }
            res.append("\n");
        }
        return res.toString();
    }

    public void reiniciar() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                conductor[i][j] = false;
            }
        }
    }
}
