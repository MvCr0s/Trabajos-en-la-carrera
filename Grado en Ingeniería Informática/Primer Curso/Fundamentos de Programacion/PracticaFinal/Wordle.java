import java.util.Scanner;
import java.io.*;

public class Wordle{
    public static void main(String[] args) throws FileNotFoundException {
    int NumIntentos=5; //Numero de intentos que tiene el jugador
    int RecorrePal=0;
    boolean Verificar0;
    boolean Verificar1;
    boolean Verificar2;
    Scanner in=new Scanner(System.in); 

    Scanner Fichero = new Scanner(new File("diccionarioWordle-1.txt"));
    int tam=0;
    while(Fichero.hasNextLine()){ //Nos dice el tamaño necesario del array que crearemos para meter todas las palabras del fichero
        Fichero.nextLine();
        tam++;
        }
    Fichero.close();
        
    String[] Diccionario=new String[tam]; //Creamos el diccionario 
    Diccionario=DiccionarioWordle(tam); //Metemos todas las palabras del fichero
    String palabra=Diccionario[(int)(Math.random()*Diccionario.length)]; //Escoge palabra aleatoria del diccionario
    String[] BarajaDicc=BarajaDicc(Diccionario,palabra);    //Barajea el diccionario :^)

    for(int i=0;i<NumIntentos;i++){  //Cuenta los intentos que lleva
    
        System.out.println(palabra);
        System.out.print("");
        String entrada=in.nextLine();  //Cadena de 0,1,2 introducida por el player 
        while(Verificarentrada(entrada)==false){ //Verifica la entrada del jugador
            entrada=in.nextLine();
        }
        
        do{  //Recorre las palabras del diccionario (RecorrePal)
            String palabraDic=BarajaDicc[RecorrePal];   //palabraDic ira cambiando segun avance el bucle

            Verificar0=Verificar0(entrada,palabra,palabraDic); //Llamamos a los metodos para que verifiquen
            Verificar1=Verificar1(entrada,palabra,palabraDic);
            Verificar2=Verificar2(entrada,palabra,palabraDic);
            RecorrePal++;
            
            //Queremos que lo haga mientras que haya palabras en el array o hasta que salga una palabra que verifique los 3 metodos (palabra posible)
            }while((RecorrePal < BarajaDicc.length)&&(!(Verificar0==true && Verificar1==true && Verificar2==true)));
            if(entrada.equals("22222")){ //Si la entrada es 22222 significa que esa es la palabra correcta por lo que ya ha ganado el programa
                System.out.println("He ganado :D");
                break;
            }else{
                palabra=BarajaDicc[RecorrePal-1];  //Sin el -1, que no lo teniamos puesto, siempre cogía la palabra siguiente a la correcta
            }
            if(i==4 && (!(entrada.equals("22222")))|| RecorrePal>=BarajaDicc.length){ //Si se han acabado los intentos y no ha adivinado la palabra el programa ha perdido
                System.out.println("Vaya, he perdido, no tengo palabras X)");
                System.out.println("Por curiosidad,¿Que palabra era?");
                String Palabranueva=in.nextLine();
                    if(Mirarpalabras(BarajaDicc,Palabranueva)==true){ //Queremos saber que palabra era
                        System.out.println("Ah rayos, esta me la sabia");
                        break;   
                }else{
                    System.out.println("Vaya esa no me sabia,¿Quieres añadirla?"); 
                    System.out.println("Responde: SI, si quieres añadirla, en caso contrario responda NO (o cualquier otra cosa :D )");
                    String respuesta= in.nextLine();
                    if(respuesta.toUpperCase().equals("SI")){ //Si no esta y el jugador quiere, añadimos la palabra al fichero
                        try{
                            FileWriter Ampliador = new FileWriter("ELDICCIONARIO.txt",true);
                            Ampliador.write(Palabranueva.toUpperCase());
                            Ampliador.close();
                        }
                        catch (IOException e) {
                        }
                        System.out.println("Se ha añadido la palabra");
                        break;
                    }else{
                        System.out.println("No se ha añadido la palabra");
                        break;
                    }
                }
            }
        }
    }
public static boolean Verificarentrada(String entrada){ //verifica la entrada 
    boolean Verificarentrada=true; 
    if(entrada.length()!=5){ //Comprueba que tiene longitud 5
        System.out.println("INTRODUZCA BIEN LA ENTRADA");
        Verificarentrada= false;
    }
    for(int k=0;k<entrada.length();k++) //Mira si hay algun caracter distinto de 0 1 o 2
        if(entrada.charAt(k)!= '0'){
            if(entrada.charAt(k)!= '1'){
                if(entrada.charAt(k)!= '2'){
                    System.out.println("INTRODUZCA BIEN LA ENTRADA");
                    Verificarentrada= false;
                    }
                }
            }
            return Verificarentrada;
    }

public static boolean Mirarpalabras(String[] BarajaDicc, String Palabranueva){ //Cuando pierde mira si la palabra ya esta en el fichero o no
    boolean Comprobarpal=false;
    for(int h=0;h<BarajaDicc.length;h++){
        if(BarajaDicc[h].equals(Palabranueva.toUpperCase())){
            Comprobarpal=true;
        }
    }
    return Comprobarpal;
}
    
public static boolean Verificar0(String entrada,String palabra,String palabraDic){
    boolean Verificar0=true;
    for(int i=0;i<entrada.length();i++){
        if(entrada.charAt(i)=='0'){
            for(int k=0;k<palabraDic.length();k++){
                if(palabra.charAt(i)==palabraDic.charAt(k)){
                    Verificar0=false;
                }
            }
        }
    }
    return Verificar0;
}

public static boolean Verificar1(String entrada,String palabra,String palabraDic){
    boolean Verificar1=true;
        for(int i=0;i<entrada.length();i++){
            if(entrada.charAt(i)=='1'){
                if(palabra.charAt(i)==palabraDic.charAt(i)){
                    Verificar1=false;
                }
                if(palabraDic.indexOf(palabra.charAt(i))==(-1)){
                    Verificar1=false;
                }
            }
        }
    return Verificar1;
}

public static boolean Verificar2(String entrada,String palabra,String palabraDic){
    boolean Verificar2=true;
    for(int i=0;i<entrada.length();i++){
        if(entrada.charAt(i)=='2'){
            if(palabra.charAt(i)!=palabraDic.charAt(i)){
                Verificar2=false;
            }
        }
    }
    return Verificar2;
}

public static String [] BarajaDicc(String[] Diccionario,String  palabra){ //Cada partida coloca de forma diferente las palabras del fichero en el array

    String[] BarajaDicc=new String[Diccionario.length];
    int Pos;
    for(int i=0;i<Diccionario.length;i++){  //Recorre BarajaDicc
        do{
        Pos=(int)(Math.random()*Diccionario.length);
        }while(Diccionario[Pos].equals("0"));
        BarajaDicc[i]=Diccionario[Pos];     //La idea es anular los huecos de las palabras que ya han salido
        Diccionario[Pos]="0";
    }
    return BarajaDicc;
}

public static String[] DiccionarioWordle(int tam) throws FileNotFoundException{ //Pasamos las palabras del fichero al array
        
        Scanner diccionarioWordle = new Scanner(new File("diccionarioWordle-1.txt"));
        String []Diccionario=new String[tam];
        
        int x=0;
        while(x<tam){
        Diccionario[x]=diccionarioWordle.nextLine();
        x++;
        }

        diccionarioWordle.close();
        return Diccionario;
   }
   
}
