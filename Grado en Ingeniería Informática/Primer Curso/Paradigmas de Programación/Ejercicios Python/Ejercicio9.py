with open('Listado.txt', 'r') as f: # with abre (open) y cierra (close) el fichero como si fuera un finally
    print(len(f.readlines()))


try:
    f = open('Listado.txt', 'w')
    print(len(f.readlines())) # f.flush() # para forzar el guardado en ese instante
except:a=7
# Nuestras líneas de código para actuar en caso de excepciones
finally:
# … haya o no excepción, se cierra
    f.close()