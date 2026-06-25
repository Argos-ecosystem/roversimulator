# Rover Navigator - Guia rapida en espanol

Simulador de un rover en una grilla. El rover usa un modelo LLM para planificar movimientos, pero solo ve lo que sus sensores ya descubrieron. Si el modelo se queda pegado, el sistema puede usar BFS como respaldo para buscar una ruta conocida.

---

## Que hace

- Crea una grilla configurable.
- Permite poner obstaculos fijos manualmente.
- Puede agregar obstaculos ocultos/moviles.
- El rover parte sin conocer el mapa.
- El sensor revela celdas cercanas.
- El modelo recibe la memoria del rover, no el mundo completo.
- Cada decision queda guardada con el mapa ASCII exacto que vio el modelo.
- Se puede exportar la mision completa como JSON.

---

## Como correrlo

Requiere Python 3.9+ y una API key.

```bash
cd /Users/agui1era/Desktop/AA/roverSimulator
./venv/bin/pip install flask openai
export OPENAI_API_KEY="sk-..."
./venv/bin/python3 app.py
```

Luego abrir:

```text
http://localhost:5050
```

Si existe `ROVER_PASSWORD` en `.env`, la app pide login.

---

## Comandos del rover

Los comandos son relativos a la orientacion actual del rover.

| Comando | Que hace |
|---|---|
| `F` | Avanza 1 celda hacia adelante |
| `B` | Retrocede 1 celda sin cambiar orientacion |
| `L` | Gira 90 grados a la izquierda sin moverse |
| `R` | Gira 90 grados a la derecha sin moverse |
| `P` | Pinta solo la celda actual |

El comando de pintar tambien acepta estos alias:

```text
PINTA
PINTAR
PAINT
```

El servidor los normaliza a:

```text
P
```

Importante: `P` solo pinta la casilla actual. No mueve, no gira, no escanea, no mueve obstaculos y no completa fases/mision por si solo.

---

## Que recibe el modelo en cada iteracion

En cada llamada de planificacion, el modelo recibe el contexto actual de la mision y debe responder con comandos.

Se le pasa:

- Texto de la mision escrita por el operador.
- Coordenadas de los puntos `A`, `B` y `C`, si existen.
- Si cada punto `A/B/C` ya fue visitado o no.
- Coordenada actual del rover.
- Orientacion actual del rover: `N`, `S`, `E` o `W`.
- Coordenada inicial del rover, usada como origen/home/start.
- Historial reciente de decisiones anteriores.
- Estado de mision que el mismo modelo declaro antes:
  - `phase`
  - `current_goal`
  - `next_goal`
  - `notes`
- Mapa ASCII con la memoria del rover.
- Rango/forma del sensor.
- Lista de comandos legales.
- Formato JSON obligatorio de respuesta.

Ejemplo de posicion:

```text
Rover at (row 4, col 2), facing E.
Rover STARTED at (row 0, col 0)
```

---

## Sensores y obstaculos

El modelo no recibe el mapa real completo. Recibe solo lo que el rover sabe.

La informacion de sensores y obstaculos se representa principalmente en el mapa ASCII:

| Simbolo | Significado |
|---|---|
| `R` | Posicion actual del rover |
| `A/B/C` | Puntos de referencia |
| `#` | Obstaculo conocido por la memoria del rover |
| `.` | Celda confirmada libre |
| `*` | Celda pintada |
| `?` | Celda no escaneada/desconocida |

Ejemplo:

```text
? ? ? ? ?
? . # . ?
? . R . ?
? ? . * ?
? ? ? ? A
```

En ese ejemplo:

- El rover esta en `R`.
- El modelo sabe que hay un obstaculo en `#`.
- Las celdas `.` ya fueron vistas como libres.
- La celda `*` ya fue pintada.
- Las celdas `?` siguen siendo desconocidas.

El prompt tambien informa la forma del sensor:

```text
N celdas hacia adelante
2 celdas hacia atras
1 celda hacia cada lado
```

Actualmente no se envia una lista separada tipo:

```text
obstaculos detectados este turno: [(2,4), (3,5)]
```

Eso queda implicito en el mapa ASCII con `#`, `.`, `?` y `*`.

---

## Respuesta esperada del modelo

El modelo debe responder solo JSON valido.

Ejemplo:

```json
{
  "moves": ["F", "R", "F", "P"],
  "reasoning": "avanzo hacia el punto A y pinto al llegar",
  "phase": "1/2",
  "current_goal": "llegar a A y pintar",
  "next_goal": "volver al origen",
  "notes": "A ya fue marcado con pintura",
  "done": false
}
```

Si devuelve comandos invalidos como `UP`, `DOWN` o texto raro, el servidor los filtra. Si no queda ningun comando valido y `done` no es `true`, se intenta una vez mas.

---

## Iteraciones de plan

Si `plan_iterations` es mayor que `1`, el modelo recibe su propio plan anterior para criticarlo y mejorarlo.

En esas pasadas extra se le agrega:

- Comandos propuestos antes.
- Razonamiento anterior.
- Coordenada donde terminaria si ejecuta ese plan.
- Preguntas para revisar si cruza obstaculos, si llega al objetivo y si hay giros inutiles.

La idea es que haga un primer borrador y luego lo refine.

---

## Ciclo de un paso

Para comandos de movimiento (`F` o `B`) y giro (`L` o `R`), el flujo general es:

```text
1. Obstaculos ocultos pueden moverse.
2. El sensor escanea alrededor del rover.
3. Se toma el siguiente comando del plan.
4. Si el destino ahora es un obstaculo conocido, aborta y replantea.
5. Si hay choque real con obstaculo no conocido, registra crash y replantea.
6. Si no hay problema, ejecuta el comando.
7. Escanea de nuevo desde la nueva posicion.
```

Para `P`, el flujo es distinto y mucho mas simple:

```text
1. Toma el comando P.
2. Pinta la celda actual.
3. Devuelve evento "painted".
```

No hay movimiento, giro, sensor, obstaculos ni cierre de fase en `P`.

---

## Endpoints principales

| Endpoint | Metodo | Uso |
|---|---|---|
| `/api/start` | `POST` | Inicia una mision con grilla/configuracion |
| `/api/plan` | `POST` | Pide al LLM/BFS la siguiente lista de comandos |
| `/api/step` | `POST` | Ejecuta el siguiente comando del plan |
| `/api/export` | `GET` | Exporta el estado completo de la mision |

---

## Limitaciones

- Estado global en memoria: es para uso local/single-user.
- No persiste misiones entre reinicios del servidor.
- El modelo solo ve memoria parcial, no el mundo real completo.
- Los sensores/obstaculos se pasan al modelo como mapa ASCII, no como lista estructurada separada.
- Movimiento solo en 4 direcciones, sin diagonales.
