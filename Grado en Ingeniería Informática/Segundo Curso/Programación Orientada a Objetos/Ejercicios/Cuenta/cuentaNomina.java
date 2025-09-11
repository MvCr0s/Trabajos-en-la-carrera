package Cuenta;

public class cuentaNomina extends cuenta{
	private float comision;
	private float importeNomina;
	private float interesMensual;
	private float saldo;

	
	
	public cuentaNomina(Cliente titular,float comision,float importeNomina,float interesMensual) {
		super(titular);
		this.comision=comision;
		this.importeNomina=importeNomina;
		this.interesMensual=interesMensual;
	}
	
	public float getImporteNomina() {
		return importeNomina;
	}
	
	public float getInteresMensuala() {
		return interesMensual;
	}
	
	public void reintegro(float cantidad) {
		assert(getSaldo()-getImporteNomina()>=0);
		if(saldo<0) {
			saldo-=cantidad-interesMensual;
		}else {
			saldo-=cantidad-interesMensual-interesMensual;
		}
		
	}
	
}
