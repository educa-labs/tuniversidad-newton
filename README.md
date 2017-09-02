# Newton.

Source Code del Sistema recomendador  Newton.

En carpeta pre_work se encuentran los programas para preprocesar las fuentes de datos. `newton`
es un paquete python (v`3.5.2`)

## Dependencias
Dependencias del código de entrenamiento y preprocesamiento:

* pandas
* numpy
* sklearn
* gensim
* nltk

Dependencias del código de producción (`newton`) estan en `requirements.txt`. Para instalar hacer 

```bash
pip install requirements.txt
```
## Documentación

### Modelo de aprendizaje
El modelo de aprendizaje consiste en 2 partes principales: 

La primera parte consiste en obtener un arreglo de carreras 
(ids) a partir de un puntaje psu y un área de interés. Esto se logra mediante un Random Forest entrenado con puntajes de
 años anteriores. El RF entrega las carreras con mayor probabilidad.
 
La segunda parte consiste en encontrar algunos vecinos cercanos a cada uno de los resultados de la etapa anterior. Esto se logra
mediante un Ball Tree que tiene todas las carreras de la base de datos. El balltree se construye usando
gower distance para las variables originales del dataset mas similaridad coseno obtenida mediante el modelo Word2Vec y 
los nombres de las carreras. Los valores de la similaridad coseno se encuentran guardados en una matriz para acelerar el computo.

Finalmente el output son las carreras del RF mas los vecinos cercanos. Esta pendiente filtrar segun atributos del usuario.



### Uso rápido

Para importar:
```python
from newton import *
```
Es necesario importar de esta manera ya que algunas funcionas deben estar en `__main__` para que los objetos serializados
puedan funcionar.

Para instanciar el recomendador  hacer:

```python
nw = Newton(np.array([i for i in range(1, 12)]),'forest/serialized','knn/serialized','data',3,5)
```
Y para recomendar:

```python
test_scores = [[800, 700, 0, 700, 720],
               [620, 640, 680, 0, 720]
               ]
area_id= 1 #id del area
nw.get_recs(area_id,test_scores)   
```
Eso último devuelve un `np.array (n,n_forest_results,k)` de carreras recomendadas con `n = len(test_scores)`


### Modelo de 

