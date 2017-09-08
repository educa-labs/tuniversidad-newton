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


### Clases

```python
class Newton:
```
Es el objeto que funciona como sistema recomendador. Lo que hace es manejar la interacción entre las clases `Tree` y `Forest`.

Métodos:

```python
def __init__(self, area_ids,serialized_forests,serialized_tree, 
            data_dir, n_forest_results=3, k=5):
```
Args: 
* area_ids `(np.array)` con ides de las areas.
* serialized_forests `(str)` path a carpeta con forests serializados.
* serialized_tree `(str)` path a carpeta con tree serializado
* data_dir `(str)` path a carpeta con datos.
* n_forest_results `(int)` cantidad de resultados obtenidos por el RF
* k `(int)` cantidad de vecinos cercanos calculados por el BallTree

retorna: Objeto `Newton`

```python
def get_recs(self, area_id, scores):
```
Args:
* area_id - int  id de area para recomendar.
* scores - np.array (n,5) arreglo de puntajes para recomendar.

retorna: np.array (n,n_forest_results,k) carreras recomendadas.


Atributos:

 * `area_ids (np.array)` con ides de las areas.
 * `balltree (Tree)` Ball tree para vecinos cercanos.
 * `active_forests (dict - int:Forest())`  diccionario que tiene por llave el id de un área y por valor el objeto `Forest()`
 de esa área o `None`
 * `n_forest_results (int)` Número de resultados obtenidos de una query a un `Forest`
 * `k (int)` número de vecinos cercanos pedidos al `Tree`
 * `serialized_forests (str)` path de carpeta con objetos `Forest` serializados.
  
```python
class Forest
```
Este objeto es un wrapper de la clase RandomForestClassifier de sklearn.

Métodos:

```python
def __init__(self, area_id,serialized_dir):
```
Args:
* `area_id (int)` id del área para recomendar
* `serialized_dir (str)` path al directorio con los modelos serializados.

Retorna: objeto de la clase.

```python
def query(self, points, n_results):
```
Args:

* `points (np.array, shape(n,5))` Arreglo con n puntajes a consultar con el orden `[mat,len,cie,his,nem]` 
* `n_results (int)` Número de resultados para cada puntaje.

Retorna:`np.array (n,n_results)` Arreglo de indíces de clases predecidas.

```python
def get_class(self,indexes):
```

Args:
* `indexes `(np.array(n,m)) Arreglo de índices de clases.

Retorna: `(np.array(n,m))` Arreglo con ids de carreras asociadas a las clases.


```python
class Tree
```
Esta clase es un wrapper de la clase `BallTree` de sklearn.

Métodos:
```python
def query(self, ids_points, k=5):
```

Args:
* `ids_points np.array(n)` Array con ids de las carreras a pedir los vecinos cercanos.
* `k (int)` Cantidad de vecinos por punto.

Retorna: `np.array(n,k)` Arreglo con k vecinos cercanos por cada punto de la consulta.



