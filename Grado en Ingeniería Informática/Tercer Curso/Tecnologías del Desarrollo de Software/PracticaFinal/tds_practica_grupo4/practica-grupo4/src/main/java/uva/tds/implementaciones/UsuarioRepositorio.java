package uva.tds.implementaciones;

import uva.tds.base.Usuario;
import uva.tds.base.HibernateUtil;
import uva.tds.base.Recompensa;
import uva.tds.interfaces.IUsuarioRepositorio;



import org.hibernate.HibernateException;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;



public class UsuarioRepositorio implements IUsuarioRepositorio{

    private Session getSession() {
		SessionFactory factory = HibernateUtil.getSessionFactory();
		try {
			return factory.getCurrentSession();
		} catch (HibernateException e) {
			throw new RuntimeException("Error al obtener la sesión de Hibernate", e);
		}catch(Exception e){
			throw new RuntimeException("Error inesperado al obtener la sesión", e);
		}
	}

    @Override
    public void registrarUsuario(Usuario usuario) {
        if(usuario==null) throw new IllegalArgumentException();
		if (getUsuario(usuario.getNif()) != null) {
			throw new IllegalArgumentException("El usuario con ese nif ya existe");
		}

        Session session = getSession();

		try {
			session.beginTransaction();
			usuario.setEstado(true);
			session.persist(usuario);
			session.flush();
			

		} catch (HibernateException e) {
			e.printStackTrace();
			session.getTransaction().rollback();
		} finally {
			session.close();
		}
       
    }

    
    @Override
    public Usuario getUsuario(String nif) {
		if(nif==null) throw new IllegalArgumentException();
		 if (nif.length() != 9) {
            throw new IllegalArgumentException("El dni introducido es incorrecto.");
        }
       	Session session = getSession();

		try {
			session.beginTransaction();

			Usuario usuario = session.get(Usuario.class, nif);
			return usuario;

		} catch (HibernateException e) {
			e.printStackTrace();
			session.getTransaction().rollback();
			throw new RuntimeException("Error inesperado al obtener la sesión", e);
		} finally {
			session.close();
		}
		

    }
    
        
    @Override
    public void actualizarUsuario(Usuario usuario) {
		Session session = getSession();
		if (usuario == null) {
			throw new IllegalArgumentException("El usuario no puede ser nulo");
		}
		try {
			session.beginTransaction();
			if(session.get(Usuario.class,usuario.getNif())==null){
				throw new IllegalArgumentException("El usuario con ese nif no existe");
			}
			session.merge(usuario);
			session.getTransaction().commit();
	
			

		} catch (HibernateException e) {
			e.printStackTrace();
			session.getTransaction().rollback();
		} finally {
			session.close();
		}
        
        
        
    }

    @Override
    public void eliminarUsuario(String nif) {
       
        Session session = getSession();
		
		try {	
			session.beginTransaction();
			Usuario usuario = session.get(Usuario.class, nif);
			if (usuario == null) {
				throw new IllegalArgumentException("El usuario con NIF " + nif + " no existe.");
			}
			session.delete(usuario);
			session.getTransaction().commit();

		} catch (HibernateException e) {
			e.printStackTrace();
			if (session.getTransaction().isActive()) {
				session.getTransaction().rollback(); // Reversa la transacción en caso de error
			}
			throw e; 
		} finally {
			session.close();
		}
        
    }

	/**
	 * Limpia las tablas 'USUARIOS' y 'RECOMPENSAS' en la base de datos, 
	 * eliminando todos los registros. 
     * Prepara un entorno limpio en pruebas automatizadas que interactúan 
	 * con un repositorio de usuarios y recompensas.
     * Elimina las tablas de la base de datos
	 */
	@Override
	public void clearDatabase() {
		Session session = getSession();
		session.getTransaction().begin();
		Query query = session.createSQLQuery("Truncate table USUARIOS");
		query.executeUpdate();
		
		session.close();

	}



    

}
