# Vídeos de calibración

Lista de los vídeos que hay que grabar para calibrar el análisis, y en qué
estado está cada uno.

## Cómo grabar

1. Arranca con `make app RECORD=1`. Si se ha cambiado código desde que lo
   arrancaste, reinícialo antes.
2. **Una sesión por vídeo**: añade la cámara con **"Grabar"** marcado, haz
   el vídeo y ciérrala con la ✕. Cada vídeo queda en `data/uploads/`
   (no va a git), con la fecha y la hora en el nombre.
3. **Al empezar cada vídeo**, enseña 3 segundos a la cámara un papel con su
   código (A1, B2…).
4. **En las posturas, quédate quieto unos 5 segundos.**
5. Las grabaciones no llaman a Gemini.

**Apunta y pasa:** estatura, peso de la caja, altura de la mesa, las
distancias reales de A1 y cualquier vídeo que repitas o hagas distinto.

**Material:** cinta métrica, caja de peso conocido (3–10 kg), mesa, silla
con ruedas o carro, unos 3 m de espacio libre.

## Datos apuntados

| Dato | Valor |
|---|---|
| Estatura | 1,77 m |
| Peso de la caja (mochila) | Sin pesar: se prueba en los dos extremos, 0,5 kg y 5–10 kg |
| Altura de la mesa (taburete) | ~40 cm |
| Distancias reales de A1 | — |
| Distancias de E2 (ojos-pantalla) | Sentado 40–50 cm; de pie ~30 cm |

## Estado

| Código | Vídeo | Estado |
|---|---|---|
| A1 | Distancia a la pantalla | **Repetir**: con la cámara a 1280×720 y distancias medidas con la cinta |
| A2 | Parpadeo | Hecho (10 parpadeos a propósito: 10 de 10 detectados) |
| A3 | Cuello | Hecho |
| A4 | Apartar la vista | Hecho |
| A5 | Hombros | Hecho. Resultado: no fiable con la cámara frontal; el aviso se ha retirado. **No hace falta repetirlo** |
| A6 | Levantarse | **Repetir**: fuera del encuadre al menos **40 s** (fueron 27) |
| B1–B4 | Escala, tronco, cuello y brazos, piernas | Hechos. De pie y recto, sin momentos de riesgo (la inclinación lateral solo se mide si no se está de perfil) |
| B5–B7 | Levantamientos | Hechos. B5 4/4, B7 8/8 (+1 artefacto al acercarse a la cámara al final), B6 8 (real ~6–9), vídeo de muestra 3/3 |
| C1–C5 | Empujar y arrastrar | Hechos. C1 2/3, C2 3/3, C3 3/3, remolcar con un brazo (C4) 1/3; sin falsos positivos (C5) ni levantamientos falsos junto a los empujes. Si se repiten, más lejos de la cámara (se cortaba la cabeza o los pies) |
| E1 | Varias personas (tres, libremente) | Hecho: `grabacion_2026-11-18_13-15-28.mp4`. Ver resultados abajo |
| E2 | De pie delante del portátil | Hecho: `grabacion_2026-11-18_14-38-04.mp4`. De pie se mide bien como trabajo con pantalla (distancia 33–37 cm, real ~30; cuello 45–52°, un riesgo real si se mantiene): no hace falta separarlo. Corregido que la gente que pasaba por detrás le quitara el número a la persona |
| D1 | Oficina, 35 min | Hecho: `grabacion_2026-11-18_15-02-54.mp4`. Ver resultados abajo |
| D2 | Manipulación, 5–10 min | Pendiente (opcional) |

## A. Puesto con pantalla: cámara del portátil (`Cámara 0`)

Sentado como trabajas normalmente.

| Código | Qué hacer | Duración |
|---|---|---|
| **A1 Distancia** | Mide con la cinta de los ojos a la pantalla. 15 s a **70 cm**, 15 s a **50**, 15 s a **40**, 15 s a **30** (o lo más cerca que puedas). Mirando la pantalla. **Apunta la distancia real de cada tramo.** | ~1 min |
| **A2 Parpadeo** | 1 min leyendo en la pantalla con naturalidad; después **10 parpadeos a propósito**, uno cada 2 s. | ~1 min 30 |
| **A3 Cuello** | 20 s pantalla · 20 s **teclado** · 20 s **móvil plano en la mesa** · 20 s **móvil en la mano** a la altura del pecho · 10 s **por encima** de la pantalla. | ~1 min 30 |
| **A4 Apartar la vista** | 25 s mirando **lejos** (ventana) · 25 s un **papel en la mesa** al lado · 25 s **a un lado** (unos 45°) · 15 s la pantalla. | ~1 min 30 |
| **A6 Levantarse** | 20 s sentado · **de pie** delante del portátil 20 s · **fuera del encuadre al menos 40 s** · vuelve y siéntate 20 s. | ~2 min |

## B. Cuerpo entero, posturas y cargas: móvil como cámara IP

**Colocación:** móvil en horizontal y fijo, a **2–4 m**, a la altura de la
cadera más o menos, viéndote **de lado**, de los pies a la cabeza con margen.

| Código | Qué hacer | Duración |
|---|---|---|
| **B1 Escala** | 10 s de pie recto, de lado · 10 s de frente. | 30 s |
| **B2 Tronco** | Aguanta **5 s** cada una: recto · ~20° · ~45° · ~60° · ~90°. Inclinación lateral a cada lado (5 s) y giro del tronco a cada lado (5 s). | ~1 min 30 |
| **B3 Cuello y brazos** | 5 s cada una: cabeza **abajo**, cabeza **atrás** · brazo a **45°**, **90°** y **por encima de la cabeza**. | ~1 min |
| **B4 Piernas** | 5 s cada una: sentadilla a medias (~60°) · sentadilla profunda · **a la pata coja**. | ~45 s |
| **B5 Levantar bien** | Caja **suelo → mesa → suelo**, **3 veces**: pegada al cuerpo, rodillas flexionadas, espalda recta. **3 s quieto** entre movimientos. | ~1 min |
| **B6 Levantar mal** | Igual, 3 veces, con la **espalda doblada**, la caja **lejos** y **girando el tronco** al dejarla. | ~1 min |
| **B7 Ritmo** | Sube y baja la caja seguido, unas **6 veces por minuto**, durante **2 min**. | 2 min |

## C. Empujar y arrastrar: móvil de lado

Recorrido de **3 m que cruce el encuadre de lado a lado**.

| Código | Qué hacer | Duración |
|---|---|---|
| **C1 Empujar bien** | Silla o carro empujado 3 m: **manos a la altura de la cintura**, tronco recto, brazos casi estirados. 3 veces. | ~1 min |
| **C2 Empujar mal** | Igual con las **manos bajas** (por debajo de los nudillos) y el **tronco inclinado**. 3 veces. | ~1 min |
| **C3 Arrastrar hacia atrás** | Tirando **caminando hacia atrás**, manos delante. 3 veces. | ~1 min |
| **C4 Remolcar** | Tirando **por detrás** caminando hacia delante. 3 veces (si la silla lo permite). | ~1 min |
| **C5 Control** | El mismo recorrido **sin empujar nada**, y luego **llevando la caja en brazos** (codos doblados). No debe salir ningún empuje. | ~45 s |

## D. Sesiones largas (validación)

| Código | Qué hacer | Duración |
|---|---|---|
| **D1 Oficina** | Trabajar con normalidad en el portátil, levantándote cuando lo harías. | 20–30 min |
| **D2 Manipulación** | Mover cajas y empujar la silla con naturalidad, móvil de lado. | 5–10 min |

## Resultados del 18-11-2026

Calibración hecha con los vídeos anteriores; cada valor está en el código
con su justificación.

**Levantamientos (NIOSH).**
- Solo cuenta un movimiento hecho con **las dos manos sujetando algo**: se
  mueven a la vez y recorren lo mismo (como mucho un 25 % de diferencia), a
  la misma altura (10 cm) y a una distancia estable entre ellas (30 cm).
  La NTP 477 dice que la ecuación no se aplica a levantamientos con una
  sola mano. En los 16 levantamientos reales: diferencias del 19 %, 4 cm
  y 24 cm como mucho. En E1 (tres personas hablando y moviendo cosas)
  los levantamientos pasan de 45 a 17.
- Un levantamiento interrumpido por una pausa a medio camino ya no se
  cuenta dos veces.
- Las bajadas de manos justo antes o después de un empuje (a menos de
  1,5 s) no son levantamientos: son las manos yendo a la silla o al carro.
- Con un **peso límite de 0 kg** (por ejemplo, la carga a más de 63 cm del
  cuerpo) el levantamiento queda marcado como "incremento acusado del
  riesgo" con el motivo; antes salía como "sin carga indicada".

**Empujes y arrastres.** También con **las dos manos** sobre el objeto (los
métodos de la guía, ISO 11228-2, son para ambas manos): a la misma altura
(10 cm) y a menos de 55 cm entre sí. En los once reales de C1–C4: 9 cm y
49 cm como mucho; una persona que pasaba andando por detrás en B6 contaba
como empuje (12 cm y 62 cm) y ya no.

**Posturas (REBA).**
- Solo se puntúan con el **cuerpo entero a la vista** (cadera, rodilla y
  tobillo): de pie junto al portátil, sin piernas en la imagen, el modelo
  inventaba el tronco flexionado 66–77° (REBA 10–11).
- La **inclinación lateral** del tronco y del cuello solo se mide si la
  persona no está de perfil: de perfil esa inclinación apunta a la cámara
  (la profundidad, lo que peor mide el modelo) y de pie y recto marcaba
  10–13° (REBA cuenta desde 15°); de frente, 1°.
- Si el cuello puntúa solo por estar girado o inclinado de lado, la causa
  lo dice ("cuello girado o inclinado de lado"), no "cuello flexionado 7°".

**Varias personas (E1 y B5+B7 lado a lado).**
- MediaPipe solo ve a una persona: se añadió un detector de personas
  (YOLOX-Tiny). Con B5 y B7 puestos lado a lado: 2 personas, con 4 y 8
  levantamientos, igual que cada vídeo por separado. Con una sola persona
  se usa el modelo de siempre (los resultados calibrados no cambian).
- Las personas a menos del 25 % de la altura de la imagen (muy lejos) no
  se analizan, y lo visto menos de 2 s (una silla, una mochila) no entra en
  el informe.
- Dos esqueletos solo son la misma persona si además tienen la cabeza y
  los hombros en el mismo sitio (en E2, alguien que pasaba por detrás
  quedaba "dentro" de la persona sentada y le quitaba el número), y para
  seguir a alguien de un fotograma al siguiente cuenta también el tamaño.
- **Límite conocido:** al cruzarse o al salir una persona y entrar otra,
  los números pueden intercambiarse. Probado añadir el color de la ropa:
  no mejoró (26 cambios frente a 28) y se quitó.

**Sesión larga de oficina (D1, 35 min en el portátil, con gente pasando y
charlando al fondo de un pasillo).**
- Antes de corregir: la persona sentada cambiaba de número 17 veces y su
  tiempo de pantalla quedaba repartido en tres (21 + 12 + 2 min), así que
  **no saltaba ningún aviso**; la gente del fondo sumaba 46 levantamientos.
- Un "levantamiento" no puede empezar antes de que la persona deje de
  verse y acabar cuando vuelve (salían de 30 s a 7 min): tras 1 s sin
  medir las manos, se empieza de cero. Los reales duran 0,7–12 s.
- Para seguir a cada persona cuentan la cabeza y los hombros y el ancho de
  hombros, no la caja del esqueleto: en el portátil el modelo se inventa
  las piernas fuera de la imagen y la caja saltaba de 0,3 a 1,1 de la
  altura de un fotograma a otro. Ahora: 0 cambios, una sola persona en
  pantalla 35 min, y los avisos saltan a los 20 min (descanso visual) y a
  los 30 (pausa). Los vídeos de calibración dan lo mismo que antes.
- El cuerpo solo se puntúa si, de pie, la persona ocupa al menos el 40 %
  de la altura de la imagen (lo más alto que se la vio entera en los
  últimos 10 s, para no perderla al agacharse). Las personas de la
  calibración ocupan de 0,54 a 0,97; las del fondo de D1, 0,31.
- Resultado: 0 momentos de riesgo y 0 levantamientos de la persona
  sentada; de la gente que pasa quedan 4 levantamientos y 9 momentos
  (personas andando muy cerca de la cámara y una limpiadora fregando).
- Rendimiento: 35 min analizados en 30 min (10 fotogramas por segundo,
  hasta tres personas), más rápido que el tiempo real.
