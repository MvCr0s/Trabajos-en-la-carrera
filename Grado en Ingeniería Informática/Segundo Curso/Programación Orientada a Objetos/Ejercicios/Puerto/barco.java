package Puerto;

import java.time.LocalDate;

public class barco {
		private static final int FIJO=10;
		
		private String matricula;
		private double eslora;
		private LocalDate anno;
		
		public barco(String m, double tamano, LocalDate a) {
			assert(m!=null);
			assert(tamano>0);
			assert(a!=null);
			assert(!m.isEmpty());
			matricula=m;
			eslora=tamano;
			anno=a;
		}

		public String getMatricula() {
			return matricula;
		}

		public double getEslora() {
			return eslora;
		}

		public LocalDate getAnno() {
			return anno;
		}
		
		public double getModulo() {
			return getEslora()*FIJO;
		}
}
