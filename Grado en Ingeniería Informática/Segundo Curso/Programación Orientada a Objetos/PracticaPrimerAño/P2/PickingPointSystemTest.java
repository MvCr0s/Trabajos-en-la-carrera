/**
 * @author mardedi
 * @author daniega
 */

package amazingco;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.List;

import es.uva.inf.poo.maps.GPSCoordinate;
import java.util.ArrayList;
import java.util.Arrays;

import org.junit.Before;
import org.junit.Test;




public class PickingPointsSystemTest {
	
	private PickingPointsSystem PPS = null;
	private HashMap<String, PickingPoint> pickingPointsMap=null;
	private PickingPoint p1 = null;
	private PickingPoint p2 = null;
	private PickingPoint p3 = null;
	private PickingPoint p4 = null;
			
	@Before
	public void inicio() {	
		PickingPoint pl = new PackageLocker(45,"72106262", new GPSCoordinate(0, 0)); 
		PickingPoint p2 = new PostOffice("27759048", new GPSCoordinate(1, 1));
		PickingPoint p3 = new Kiosk("27759048", new GPSCoordinate(20, 0), 20,2,true);
		PickingPoint p4 = new PackageLocker(10,"31330168", new GPSCoordinate(20, 20)); 
		
		pickingPointsMap = new HashMap<String, PickingPoint>();
		
		pickingPointsMap.put(pl.getIdentificador(), pl);
		pickingPointsMap.put(p2.getIdentificador(), p2);
		pickingPointsMap.put(p3.getIdentificador(), p3);
		pickingPointsMap.put(p4.getIdentificador(), p4);
		
		PickingPointsSystem PPS = new PickingPointsSystem(pickingPointsMap);

	}
	
	//pruebas con el constructor
	@Test
	public void iniciocorrectamente() {    
		PickingPoint pl = new PackageLocker(45,"72106262", new GPSCoordinate(0, 0)); 
		PickingPoint p2 = new PostOffice("27759048", new GPSCoordinate(1, 1));
		PickingPoint p3 = new Kiosk("27759048", new GPSCoordinate(20, 0), 20,2,true);
		PickingPoint p4 = new PackageLocker(10,"31330168", new GPSCoordinate(20, 20));  
	    
	    pickingPointsMap.put(pl.getIdentificador(), pl);
	    pickingPointsMap.put(p4.getIdentificador(), p4);
	    pickingPointsMap.put(p3.getIdentificador(), p3);
	    pickingPointsMap.put(p2.getIdentificador(), p2);
	    
	    assertEquals(pl, PPS.getHashMap().get("72106262"));
	    assertEquals(p2, PPS.getHashMap().get("27759048"));
	    assertEquals(p3, PPS.getHashMap().get("61202998"));
	    assertEquals(p4, PPS.getHashMap().get("31330168"));
	}
	
	@Test
	public void testPickingPointsSystemConstruirVacio() {
		PickingPointsSystem PPSnull = new PickingPointsSystem(null);
		assertTrue(PPSnull.getHashMap().isEmpty());
		}
	
	@Test
	public void testPickingPointsSystemSetHashMap() {
		HashMap<String, PickingPoint> pickingPointsMap2 = new HashMap<String, PickingPoint>();
		PickingPointsSystem PPS2 = new PickingPointsSystem();
		
		PickingPoint p2 = new PostOffice("27759048", new GPSCoordinate(1, 1));
		pickingPointsMap2.put(p2.getIdentificador(), p2);
		PPS2.setHashMap(pickingPointsMap2);
	    assertEquals(PPS2.getHashMap(), p2º);
	}
//_______________________________________________________________________________________________


	@Test
	public void testPickingPointsSystemGetDistancetoPickingPoint() {

	    ArrayList<PickingPoint> pickingPointList = new ArrayList<>();
	    pickingPointList.add(p1);
	    pickingPointList.add(p2);
	    pickingPointList.add(p3);
	    pickingPointList.add(p4);
	    assertEquals(PPS.getDistancetoPickingPoint(50, new GPSCoordinate(0, 0)), pickingPointList);
	}

	@Test
	public void testPickingPointsSystemGetDistancetoPickingPointFarAway() {
	    ArrayList<PickingPoint> pickingPointList = new ArrayList<>();
	    pickingPointList.add(p1);
	    pickingPointList.add(p2);
	    pickingPointList.add(p3);
	    pickingPointList.add(p4);

	    assertNotEquals(PPS.getDistancetoPickingPoint(10, new GPSCoordinate(80, 150)), pickingPointList);
	}
	

	@Test(expected = IllegalArgumentException.class)
	public void testPickingPointsSystemGetDistancetoPickingPointNoAcceptableGPSCoordinates() {
	    HashMap<String, PickingPoint> pickingPointsSystemMap = new HashMap<>();
	    GPSCoordinate coordinates = new GPSCoordinate(10000, 1000);
	}

	@Test
	public void testAddPickingPoint() {
		PickingPoint p5 = new PackageLocker(45,"72106262", new GPSCoordinate(0, 0)); 
		PPS.addPickingPoint(p5);
	    assertTrue(PPS.getHashMap().containsKey(p5.getIdentificador()));
	    assertTrue(PPS.getHashMap().containsValue(p5));
	}

	public void testremovePickingPoint() {
	    PPS.removePickingPoint("72106262C");
	    assertTrue(PPS.getHashMap().containsKey(p1.getIdentificador()));
	    assertTrue(PPS.getHashMap().containsValue(p1));
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testremovePickingPointIAE() {
	    PPS.removePickingPoint("12345");
	}

	@Test
	public void testGetPickingPointListoperativos() {
	    ArrayList<PickingPoint> lockerListOperativos = new ArrayList<>();
	    lockerListOperativos.add(p1);
	    lockerListOperativos.add(p2);
	    lockerListOperativos.add(p3);
	    lockerListOperativos.add(p4);	    
	    assertEquals(PPS.getPickingPointListOperativos(), lockerListOperativos);
	}
	
	public void testGetPickingPointListoperativonull () {
	    p2.setstate(false);
	    p1.setstate(false);
	    p4.setstate(false);
	    p3.setstate(false);
	    ArrayList<PackageLocker> lockerListOperativos = new ArrayList<>();
	    assertNotEquals(PPS.getPickingPointListOperativos(), lockerListOperativos);
	}
	
	
	@Test
	public void testGetPickingPointListoutofservice() {
	    p4.setstate(false);
	    p3.setstate(false);
	    
		ArrayList<PickingPoint> LockerListNoOperativos = new ArrayList<>();
		LockerListNoOperativos.add(p3);
	    LockerListNoOperativos.add(p4);
		assertEquals(PPS.getPickingPointListOperativos(),LockerListNoOperativos);
		
	}

	@Test
	public void testGetPickingPointListoutofservicenull() {
	    p2.setstate(true);
	    p1.setstate(true);
	    p4.setstate(true);
	    p3.setstate(true);
	    ArrayList<PackageLocker> LockerListNoOperativos = new ArrayList<PackageLocker>();
	    assertNotEquals(PPS.getPickingPointListOperativos(), LockerListNoOperativos);
	}
	
	
	@Test
	public void testGetPickingPointListempty() {
		PackageLocker p5 = new PackageLocker(5,"3133016", new GPSCoordinate(2, 20)); 
		p5.settaquillasvacias(0);
		PPS.addPickingPoint(p5);
		ArrayList<PickingPoint> LockerListempty = new ArrayList<>();
		LockerListempty.add(p1);
		LockerListempty.add(p2);
		LockerListempty.add(p3);
		LockerListempty.add(p4);
		assertEquals(PPS.getPickingPointListEmpty(),LockerListempty);
	}
	
	
	@Test
	public void testGetPackageLockerListemptynull() {
		PickingPointsSystem PPS2 = new PickingPointsSystem();
		PackageLocker p5 = new PackageLocker(5,"31330168", new GPSCoordinate(2, 20)); 
		PPS2.addPickingPoint(p5);
		p5.settaquillasvacias(0);
		assertEquals(PPS2.getPickingPointListEmpty(),null);
	}

}
