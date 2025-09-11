package com.example;


import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.easymock.EasyMock;
import org.easymock.Mock;
import org.easymock.TestSubject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import uva.tds.Domotica;
import uva.tds.DomoticaService;


public class DomoticaTest{

    @TestSubject
    private Domotica app;

    @Mock 
    private DomoticaService service;

    
    @BeforeEach
    public void starUp(){
        
        service=EasyMock.mock(DomoticaService.class);

    }

    @Test
    public void constructorValido(){
        app= new Domotica(service);
        assertNotNull(app);
        assertEquals(service, app.getDomoticaService());
    }

    public void constructorInvalido(){
        assertThrows(IllegalArgumentException.class , ()->{ new Domotica(null);} );
    }
    @Test
    public void initServicioNoNull(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(1);
        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        assertTrue(app.isAvailable());
        EasyMock.verify(service);

    }

    @Test
    public void initServicioEncenderLuces(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(2);

        service.encenderLuces(24);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.lucesEncencidas()).andReturn(true).times(1);
        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        app.encenderLuces(24);
        assertTrue(app.lucesEncencidas());
        EasyMock.verify(service);

    }

    
    @Test
    public void initServicioEncenderLucesMayor(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(1);

        service.encenderLuces(101);
        EasyMock.expectLastCall().andThrow(new IllegalArgumentException());
        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        assertThrows(IllegalArgumentException.class,()->{app.encenderLuces(101); } );
        EasyMock.verify(service);

    }


    @Test
    public void initServicioEncenderLucesMenor(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(1);

        service.encenderLuces(-1);
        EasyMock.expectLastCall().andThrow(new IllegalArgumentException());
        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        assertThrows(IllegalArgumentException.class,()->{app.encenderLuces(-1); } );
        EasyMock.verify(service);

    }

    @Test
    public void initServicioEncenderLucesSinHabilitarse(){

        EasyMock.expect(service.isAvailable()).andReturn(false).times(1);

        EasyMock.replay(service);

        app=new Domotica(service);
        assertThrows(IllegalStateException.class,()->{app.encenderLuces(43); } );
        EasyMock.verify(service);

    }

    @Test
    public void initServicioEncenderLucesyaEncendidas(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(2);

        service.encenderLuces(24);
        EasyMock.expectLastCall().times(1);

        service.encenderLuces(24);
        EasyMock.expectLastCall().andThrow(new IllegalStateException());
        
        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        app.encenderLuces(24);
        assertThrows(IllegalStateException.class,()->{app.encenderLuces(24); } );
        EasyMock.verify(service);

    }

    @Test
    public void initServicioApagarLuces(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(3);

        service.encenderLuces(24);
        EasyMock.expectLastCall().times(1);

        service.apagarLuces();
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.lucesEncencidas()).andReturn(false).times(1);

        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        app.encenderLuces(24);
        app.apagarLuces();
        assertFalse(app.lucesEncencidas());
        EasyMock.verify(service);

    }

    @Test
    public void initServicioApagarLucesYaApagadas(){
        service.setAvailable(true);
        EasyMock.expectLastCall().times(1);
        EasyMock.expect(service.isAvailable()).andReturn(true).times(3);

        service.encenderLuces(24);
        EasyMock.expectLastCall().times(1);

        service.apagarLuces();
        EasyMock.expectLastCall().times(1);

        service.apagarLuces();
        EasyMock.expectLastCall().andThrow(new IllegalStateException());

        EasyMock.replay(service);

        app=new Domotica(service);
        app.init();
        app.encenderLuces(24);
        app.apagarLuces();
        assertThrows(IllegalStateException.class,()->{app.apagarLuces(); } );
        EasyMock.verify(service);

    }


    @Test
    public void initServicioApagarLucesSinHabilitar(){
        EasyMock.expect(service.isAvailable()).andReturn(false).times(1);

        EasyMock.replay(service);

        app=new Domotica(service);
        assertThrows(IllegalStateException.class,()->{app.apagarLuces(); } );
        EasyMock.verify(service);

    }


    @Test
    public void initServicioLucesEncendidasSinHabilitar(){
        EasyMock.expect(service.isAvailable()).andReturn(false).times(1);

        EasyMock.replay(service);

        app=new Domotica(service);
        assertThrows(IllegalStateException.class,()->{app.lucesEncencidas(); } );
        EasyMock.verify(service);

    }

}