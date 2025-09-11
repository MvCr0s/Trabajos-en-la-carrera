package Aeropuerto;

import java.time.LocalTime;

import static org.junit.Assert.*;
import org.junit.Test;

public class nuevoVuelo{
	private String flightCode;
	private String oprigin;
	private String destino;
	private LocalTime time;
	
	

		@Test
		public void insertFlight() {
			insertNewFlight("AI3434","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight() {
			insertNewFlight("AI343A","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight2() {
			insertNewFlight("AIA434","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight3() {
			insertNewFlight("Ai2434","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight4() {
			insertNewFlight("AI243","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight5() {
			insertNewFlight("AI24333","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
			
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight6() {
			insertNewFlight("      ","Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmalcodFlight7() {
			insertNewFlight(null,"Madrid", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmaloriginFlight() {
			insertNewFlight("AI2433"," ", "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmaloriginFlight2() {
			insertNewFlight("AI2433",null, "Londres", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmaldestinoFlight() {
			insertNewFlight("AI2433","Madrid", " ", LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmaldestinoFlight2() {
			insertNewFlight("AI2433","Madrid",null, LocalTime.of(16, 35, 0));
		}
		
		@Test (expected = IllegalArgumentException.class)
		public void insertmaltimeFlight() {
			insertNewFlight("AI2433","Madrid","Londres", null);
		}
				
		
		
condicion 								clases de equivalencia					clases de equivalencia Novalidas
Long de cadena=6                        1.1-Long 6           					1.2-Long<6  1.3-Long>6
1-2c son letras							2.1-'A'<=c<='Z' 						2.2-c<'A'   2.3-c>'Z'
2-6c son numeros						3.1-'0'<=c<='9'							3.2-c<'0'   3.3-c>'9'
origin sea origen del vuelo				4.1-origin sea ciudad de origen			4.2-origin no se ciudad de origen
dest sea destino del vuelo				5.1-dest sea ciudad de destino			5.2-dest no se ciudad de destino
time del vuelo							6.1-hora del vuelo						6.2- null



prueba(valida)											Clase de equivalencia			Salida esperada
"AE1234","Madrid","Londres",LocalTime.of(16, 35, 0)		1.1 2.1 3.1 4.1 5.1


Prueba(invalida)			Clases de equivalencia					Salida esperada
"AE123"							1.2							@IllegalArgumentException
"AE12345"						1.3							@IllegalArgumentException
"A012345"						2.2							@IllegalArgumentException
"A|12345"						2.3							@IllegalArgumentException
"A0.2345"						3.2							@IllegalArgumentException
"AEE2345"						3.3							@IllegalArgumentException
origen:null						4.2							@IllegalArgumentException
destino:null					5.2							@IllegalArgumentException
LocalTime.of(null)				6.2							@IllegalArgumentException



}
