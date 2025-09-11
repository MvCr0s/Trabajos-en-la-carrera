import java.util.*;

class Asignatura {

    private int codigo;
    private String nombre;
    private int creditos;
    private String acronimo;
    private DuracionAsignatura duracion;
    private TipoAsignatura tipo;
    private AbstractCollection<PlanDeEstudios> planes;

    public Asignatura() {
        // TODO
        planes = new ArrayList<PlanDeEstudios>();
        assert(check1());
        assert(check2());
        assert(check3());
    }

    public boolean check1() {
        // TODO inv:self.esCuatrimestral=trueimplies(self.AsignaturaEnPlan.periodo=Periodo::PRIMERCUATRIMESTREorself.AsignaturaEnPlan.periodo=Periodo::SEGUNDOCUATRIMESTRE)
        return false;
    }

    public boolean check2() {
        // TODO inv:self.esCuatrimestral=falseimplies(self.AsignaturaEnPlan.periodo=Periodo::ANUAL)
        return false;
    }

    public boolean check3() {
        // TODO inv:self.AsignaturaEnPlan.AsignacionAreaAsignaturaEnPlan.porcentaje->sum()=100
        return false;
    }

    public int getCodigo() {
        return codigo;
    }

    public void setCodigo(int codigo) {
        this.codigo = codigo;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public int getCreditos() {
        return creditos;
    }

    public void setCreditos(int creditos) {
        this.creditos = creditos;
    }

    public String getAcronimo() {
        return acronimo;
    }

    public void setAcronimo(String acronimo) {
        this.acronimo = acronimo;
    }

    public DuracionAsignatura getDuracion() {
        return duracion;
    }

    public void setDuracion(DuracionAsignatura duracion) {
        this.duracion = duracion;
    }

    public TipoAsignatura getTipo() {
        return tipo;
    }

    public void setTipo(TipoAsignatura tipo) {
        this.tipo = tipo;
    }

}

class Profesor {

    private String numeroDeDespacho;
    private String telefonoDespacho;

    public Profesor() {
        // TODO
        assert(check1());
        assert(check2());
        assert(check3());
    }

    public boolean check1() {
        // TODO inv:self.ProfesorPreviamenteEnArea.dedicacionEnPeriodo>=0andself.ProfesorPreviamenteEnArea.dedicacionEnPeriodo<=100
        return false;
    }

    public boolean check2() {
        // TODO inv:self.grupos.asignatura.periodo=Periodo::PRIMERCUATRIMESTREimpliesself.horas->size()=(6*(self.ProfesorPreviamenteEnArea.dedicacionEnPeriodo/100))
        return false;
    }

    public boolean check3() {
        // TODO inv:self.grupos.asignatura.periodo=Periodo::SEGUNDOCUATRIMESTREimpliesself.horas->size()=(6*(self.ProfesorPreviamenteEnArea.dedicacionEnPeriodo/100))
        return false;
    }

    public String getNumeroDeDespacho() {
        return numeroDeDespacho;
    }

    public void setNumeroDeDespacho(String numeroDeDespacho) {
        this.numeroDeDespacho = numeroDeDespacho;
    }

    public String getTelefonoDespacho() {
        return telefonoDespacho;
    }

    public void setTelefonoDespacho(String telefonoDespacho) {
        this.telefonoDespacho = telefonoDespacho;
    }

}

class asignacion {


    public asignacion() {
        // TODO
    }

}

class Empleado {

    private String numeroDeEmpleado;
    private String correoInterno;
    private String IBAN;

    public Empleado() {
        // TODO
    }

    public String getNumeroDeEmpleado() {
        return numeroDeEmpleado;
    }

    public void setNumeroDeEmpleado(String numeroDeEmpleado) {
        this.numeroDeEmpleado = numeroDeEmpleado;
    }

    public String getCorreoInterno() {
        return correoInterno;
    }

    public void setCorreoInterno(String correoInterno) {
        this.correoInterno = correoInterno;
    }

    public String getIBAN() {
        return IBAN;
    }

    public void setIBAN(String IBAN) {
        this.IBAN = IBAN;
    }

}

class Persona {

    private String nif;
    private String nombre;
    private String apellidos;
    private String telefono;
    private String correo;
    private Direccion direccion;

    public Persona() {
        // TODO
    }

    public String getNif() {
        return nif;
    }

    public void setNif(String nif) {
        this.nif = nif;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public String getApellidos() {
        return apellidos;
    }

    public void setApellidos(String apellidos) {
        this.apellidos = apellidos;
    }

    public String getTelefono() {
        return telefono;
    }

    public void setTelefono(String telefono) {
        this.telefono = telefono;
    }

    public String getCorreo() {
        return correo;
    }

    public void setCorreo(String correo) {
        this.correo = correo;
    }

    public Direccion getDireccion() {
        return direccion;
    }

    public void setDireccion(Direccion direccion) {
        this.direccion = direccion;
    }

}

class AreaDeConocimiento {

    private String siglas;
    private String nombre;

    public AreaDeConocimiento() {
        // TODO
    }

    public String getSiglas() {
        return siglas;
    }

    public void setSiglas(String siglas) {
        this.siglas = siglas;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

}

class PlanDeEstudios {

    private int codigo;
    private NivelEstudio nivel;
    private String idiomas;
    private int creditos;
    private TipoEnsennanza tipo;
    private int duracion;
    private int plazas;

    public PlanDeEstudios() {
        // TODO
    }

    public int getCodigo() {
        return codigo;
    }

    public void setCodigo(int codigo) {
        this.codigo = codigo;
    }

    public NivelEstudio getNivel() {
        return nivel;
    }

    public void setNivel(NivelEstudio nivel) {
        this.nivel = nivel;
    }

    public String getIdiomas() {
        return idiomas;
    }

    public void setIdiomas(String idiomas) {
        this.idiomas = idiomas;
    }

    public int getCreditos() {
        return creditos;
    }

    public void setCreditos(int creditos) {
        this.creditos = creditos;
    }

    public TipoEnsennanza getTipo() {
        return tipo;
    }

    public void setTipo(TipoEnsennanza tipo) {
        this.tipo = tipo;
    }

    public int getDuracion() {
        return duracion;
    }

    public void setDuracion(int duracion) {
        this.duracion = duracion;
    }

    public int getPlazas() {
        return plazas;
    }

    public void setPlazas(int plazas) {
        this.plazas = plazas;
    }

}

class AsignaturaEnPlan {

    private Set<AreaDeConocimiento> areasDeConocimientoAsignadas;

    public AsignaturaEnPlan() {
        // TODO
        areasDeConocimientoAsignadas = new HashSet<AreaDeConocimiento>();
        assert(check1());
        assert(check2());
        assert(check3());
        assert(check4());
        assert(check5());
        assert(check6());
    }

    public boolean check1() {
        // TODO inv:(self.tipo=TipoAsignatura::BASICAorself.tipo=TipoAsignatura::OBLIGATORIA)impliesself.grupos.tipo->select(t|t=TipoGrupo::TEORIA)->size()>=1
        return false;
    }

    public boolean check2() {
        // TODO inv:(self.tipo=TipoAsignatura::BASICAorself.tipo=TipoAsignatura::OBLIGATORIA)impliesself.grupos.tipo->select(t|t=TipoGrupo::PRACTICAS)->size()>=1
        return false;
    }

    public boolean check3() {
        // TODO inv:self.tipo=TipoAsignatura::OPTATIVAand(self.grupos.curso.fechadeInicio.getYear().isEqual(LocalDate.Now.getYear()-1)andself.grupos.estudiantes->size()<5)implies(self.grupos.curso.fechadeInicio.getYear().isEqual(LocalDate.Now.getYear())andself.grupos->size()>=0)
        return false;
    }

    public boolean check4() {
        // TODO inv:self.periodo=Periodo::ANUALimplies(self.grupos.profesores->size()>0and(LocalDate.Now.isBefore(self.grupos.curso.fechadeInicio)))
        return false;
    }

    public boolean check5() {
        // TODO inv:self.periodo=Periodo::PRIMERCUATRIMESTREimplies(self.grupos.profesores->size()>0and(LocalDate.Now.isBefore(self.grupos.curso.fechadeInicio)))
        return false;
    }

    public boolean check6() {
        // TODO inv:self.periodo=Periodo::SEGUNDOCUATRIMESTREimpliesself.grupos.profesores->size()>0and(LocalDate.Now.isBefore(self.grupos.curso.fechadeInicioSegundoCuatrimestre))
        return false;
    }

}

public enum NivelEstudio {
    GRADO,
    MASTER,
    DOCTORADO
}

public enum DuracionAsignatura {
    CUATRIMESTRAL,
    ANUAL
}

public enum TipoGrupo {
    TEORIA,
    PRACTICA,
    SEMINARIO,
    AULA
}

public enum TipoEnsennanza {
    PRESENCIAL,
    SEMIPRESENCIAL,
    VIRTUAL
}

public enum TipoDeVia {
    CALLE,
    AVENIDA,
    PLAZA,
    PASEO,
    CARRETERA
}

public enum Periodo {
    PRIMERCUATRIMESTRE,
    SEGUNDOCUATRIMESTRE,
    ANUAL
}

public enum TipoAsignatura {
    OBLIGATORIA,
    BASICA,
    OPTATIVA
}

record Direccion(TipoDeVia tipoDeVia, String nombreDeLaVia, int numero, String otros, int codigoPostal, String localidad, String Provincia) {}

record LocalDate(int day, int month, int year) {}

