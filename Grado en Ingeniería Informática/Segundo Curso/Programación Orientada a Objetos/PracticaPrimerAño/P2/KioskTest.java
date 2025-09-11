package PackageLocker;

import static org.junit.Assert.*;


import java.util.HashMap;
import java.util.List;

import es.uva.inf.poo.maps.GPSCoordinate;

import java.time.LocalDate;
import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import com.amazingco.After;
import com.amazingco.Kiosk;
import com.amazingco.Package;

public class KioskTest {
	
	protected Kiosk kiosk = null;

	@Before
	public void beforeTest() {

		kiosk = new Kiosk("72106262C", new GPSCoordinate(3, 33), 5,0,true);
		Package p1 = new Package(LocalDate.of(2024, 1 ,20), "1111111119", 7.50,true);
		Package p2 = new Package(LocalDate.of(2024, 2, 20), "0000000000", 10.0,false);
		kiosk.addPackage(p1);
		kiosk.addPackage(p2);
	}

	@After
	public void afterTest() {
		kiosk = null;
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void anadirPaqueteaKioskSinEspacio() {
		kiosk.setStorageCapacity(3);
		Package p3 = new Package(LocalDate.of(2024, 2, 20), "1111111119", 10.0,false);
		Package p4 = new Package(LocalDate.of(2024, 2, 20), "0000000000", 10.0,false);
		kiosk.addPackage(5, p3);
		
		assertEquals(kiosk.getStorageCapacity(),3);
		assertEquals(kiosk.getPackageCertified(p3.getpackagecode()),p3.getDNI());
		assertEquals(kiosk.getStorageCapacity(),3);
		assertFalse(kiosk.getPackageCertified(p4.getpackagecode()),p4.getDNI());
		kiosk.addPackage(p4);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void kioskStorageCapacityCero() {
		Kiosk k2 = new Kiosk("18090996R",new GPSCoordinate(3424,234), 0,0,true);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void kioskStorageCapacityMenosUno() {
		Kiosk k2 = new Kiosk("18090996R",new GPSCoordinate(3424,234), -1,0,true);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void anadirPaqueteQueYaEstá() {
		Package p3 = new Package(LocalDate.of(2024, 2, 20), "1111111119", 10.0,false);
		kiosk.addPackage(p3);;
	}
	
	
	@Test(expected = IllegalArgumentException.class)
	public void setMenosStorageCapacity() {
		kiosk.setStorageCapacity(4);
		Package p3 = new Package(LocalDate.of(2024, 2, 20), "1111111119", 10.0,false);
		Package p4 = new Package(LocalDate.of(2024, 2, 20), "0000000000", 10.0,false);
		kiosk.addPackage(p3);;
		kiosk.addPackage(p4);
		assertEquals(kiosk.getPackageCertified(p3.getpackagecode(),p3.getDNI()),p3);
		assertEquals(kiosk.getPackageCertified(p4.getpackagecode()),p4.getDNI()),p4);
		assertEquals(kiosk.getStorageCapacity(),4);
		assertEquals(kiosk.getllenos(),2);
		kiosk.setStorageCapacity(1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void setStorageCapacityNegativa() {
		Kiosk kiosk2 = new Kiosk("0000000000",new GPSCoordinate(33, 33), 1,0,true);
		Package p0 = new Package(LocalDate.of(2024, 2, 20), "1111111119", 10.0,false);;
		kiosk2.addPackage(p0);;
		assertTrue(kiosk2.getPackageCertified(0, "1111111119"));
		assertEquals(kiosk2.getllenos(),1);
		kiosk2.setStorageCapacity(0);
	}
	
	@Test
	public void recogerPackageNoCertified() {
		kiosk.setCashToDeliver(0);
		assertEquals(kiosk.getCashToDeliver(),0);
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,false ,"1111111119");
		Package p4 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,false ,"0000000000");

		kiosk.addPackage(p3);
		kiosk.addPackage(p4);
		assertEquals(kiosk.getCashToDeliver(),00.0);
		assertEquals(kiosk.getPackageNoCertified(p3.getpackagecode(),p3);
		assertEquals(kiosk.getCashToDeliver(),15.0);
		assertEquals(kiosk.getPackageNoCertified(p4.getpackagecode(),p4);
		kiosk.deliverCashToAmazingCo();
		assertEquals(kiosk.getCashToDeliver(),30.0);
		kiosko.deliverCashToAmazingCo();
		assertEquals(kiosk.getCashToDeliver(),00.0);
	}
	
	public void recogerPackageCertified() {
		kiosk.setCashToDeliver(0);
		assertEquals(kiosk.getCashToDeliver(),0);
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,true ,"1111111119");
		Package p4 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,true ,"0000000000");
		kiosk.addPackage(p3);
		kiosk.addPackage(p4);
		assertEquals(kiosk.getCashToDeliver(),0.0);
		assertEquals(kiosk.getPackageCertified(p3.getpackagecode(),p3.getDNI(),p3);
		assertEquals(kiosk.getCashToDeliver(),15.0);
		assertEquals(kiosk.getPackageCertified(p4.getpackagecode(),p3.getDNI(),p4);
		kiosk.deliverCashToAmazingCo();
		assertEquals(kiosk.getCashToDeliver(),30.0);
		kiosko.deliverCashToAmazingCo();
		assertEquals(kiosk.getCashToDeliver(),00.0);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void DineroNegativo() {
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,true ,"0000000000");
		kiosk.addPackage(p3);
		kiosk.getPackageNoCertified(p3.getpackagecode(),p3.getDNI());
		kiosk.setCashToDeliver(-1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void PackageCertifiedInexistente() {
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,true ,"0000000000");
		kiosk.getPackageCertified(p3.getpackagecode(), p3.getDNI());
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void PackageNoCertifiedInexistente() {
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,false ,"0000000000");
		kiosk.getPackageNoCertified(p3.getpackagecode());
	}
	@Test(expected = IllegalArgumentException.class)
	public void devolverPaqueteDNIincorrecto() {
		Package p3 = new Package(LocalDate.of(2024, 2, 20),false ,15.00,true ,"1111111119");
		kiosk.addPackage(p3);
		kiosk.getPackageCertified(p3.getpackagecode(),"0000000000");

	}
	@Test(expected = IllegalArgumentException.class)
	public void crearKioskLocalizacionNulo() {
		Kiosk kiosk2 = new Kiosk("0000000000",null, 1,0,true);
	}
	
	public void AvalaibleSpace() {
		Kiosk kiosk2 = new Kiosk("0000000000",new GPSCoordinate(3, 33), 1,0,true);;
		Package p3 = new Package(LocalDate.of(2024, 2, 20),true ,15.00,true ,"0000000000");
		kiosk2.addPackage(p3);
		kiosk.setllenos(1);
		Package p4 = new Package(LocalDate.of(2024, 2, 20),true ,15.00,true ,"0000000000");
		if(kiosk.hasAvailableSpace()>0) {
			kiosk.addPackage(p4);
		}

	}
	

}
