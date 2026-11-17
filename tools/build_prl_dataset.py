"""Build the evaluation dataset of the prevention corpus.

Each question names the corpus file that answers it and a literal quote
of the passage: the source is found by that quote, so the dataset is
rebuilt as it is whenever the corpus is (no character ranges written by
hand). When other files give the same words (the INSST guide reproduces
the articles of RD 487/1997), they are sources too: evaluate with
``--any_source``. The source is the whole paragraph of the quote, widened to at
least ``MIN_SOURCE`` characters (a cell of a table is a paragraph of a
few words).

Writes the questions in the format of the original project, for
``search_dataset`` and ``evaluate``:
    data/datasets/prl/UnansweredQuestions/dataset_prl.json
    data/datasets/prl/AnsweredQuestions/dataset_prl.json
and the same for ``dataset_prl_holdout.json``: questions on passages the
first dataset does not use, kept apart to check that a change tuned on
the first one is real. Nothing is ever tuned on the holdout.

From the project root, after ``tools/build_prl_corpus.py``:
    uv run python tools/build_prl_dataset.py
"""
import json
from pathlib import Path
import sys

CORPUS = Path("data/raw/prl")
OUT = Path("data/datasets/prl")
MIN_SOURCE = 300

# (id, question, file stem, literal quote of the answer, answer)
QUESTIONS = [
    ("rd-definicion",
     "¿Qué se entiende legalmente por manipulación manual de cargas?",
     "rd_487_1997_manipulacion_cargas",
     "se entenderá por manipulación manual de cargas cualquier operación de "
     "transporte o sujeción de una carga",
     "Cualquier operación de transporte o sujeción de una carga por uno o "
     "varios trabajadores que entrañe riesgos, en particular dorsolumbares."),
    ("rd-evitar",
     "¿Qué tiene que hacer la empresa para que los trabajadores no tengan "
     "que levantar cargas a mano?",
     "rd_487_1997_manipulacion_cargas",
     "El empresario deberá adoptar las medidas técnicas u organizativas "
     "necesarias para evitar la manipulación manual de las cargas",
     "Adoptar medidas técnicas u organizativas para evitarla, en especial "
     "equipos para el manejo mecánico."),
    ("rd-evaluar",
     "Si no se puede evitar levantar cargas a mano, ¿qué tiene en cuenta la "
     "empresa al evaluar el riesgo?",
     "rd_487_1997_manipulacion_cargas",
     "deberá evaluar los riesgos tomando en consideración los factores "
     "indicados en el anexo",
     "Los factores del anexo del Real Decreto y sus efectos combinados."),
    ("rd-formacion",
     "¿Qué formación tienen que dar a quien manipula cargas?",
     "rd_487_1997_manipulacion_cargas",
     "proporcionará a los trabajadores una formación e información adecuada "
     "sobre la forma correcta de manipular las cargas",
     "Formación e información sobre la forma correcta de manipular las "
     "cargas y los riesgos de no hacerlo."),
    ("rd-vigilancia",
     "¿Quién hace los reconocimientos médicos a los trabajadores que "
     "levantan cargas?",
     "rd_487_1997_manipulacion_cargas",
     "Tal vigilancia será realizada por personal sanitario competente",
     "Personal sanitario competente, según los protocolos de las "
     "autoridades sanitarias."),
    ("rd-carga",
     "¿Cuándo es peligrosa una carga por sus propias características?",
     "rd_487_1997_manipulacion_cargas",
     "Cuando la carga es demasiado pesada o demasiado grande.",
     "Cuando es demasiado pesada o grande, voluminosa, inestable..."),
    ("rd-torsion",
     "¿Es peligroso levantar una carga girando o doblando el tronco?",
     "rd_487_1997_manipulacion_cargas",
     "Cuando no puede realizarse más que por un movimiento de torsión o de "
     "flexión del tronco.",
     "Sí: el esfuerzo físico entraña riesgo dorsolumbar en ese caso."),
    ("rd-80kg",
     "¿Qué normas antiguas sobre sacos de 80 kilos quedaron derogadas?",
     "rd_487_1997_manipulacion_cargas",
     "Quedan derogados el Decreto del Ministerio de Trabajo de 15 de "
     "noviembre de 1935",
     "El Decreto de 15 de noviembre de 1935 y la Orden de 2 de junio de "
     "1961."),
    ("guia-3kg",
     "¿A partir de qué peso puede ser un riesgo levantar un objeto?",
     "guia_tecnica_mmc_2024",
     "objetos que igualen o superen los 3 kg pueden ser fuente de riesgo",
     "A partir de 3 kg."),
    ("guia-25kg-hombre",
     "¿Cuál es el peso máximo para un hombre de 30 años en condiciones "
     "ideales?",
     "guia_tecnica_mmc_2024",
     "25 kg en el caso de hombres de 20 a 45 años",
     "25 kg."),
    ("guia-mas-25kg",
     "¿Levantar más de 25 kilos es un riesgo por sí mismo?",
     "guia_tecnica_mmc_2024",
     "la MMC con un peso superior a 25 kg puede constituir por sí misma un "
     "riesgo no tolerable",
     "Sí, puede constituir por sí mismo un riesgo no tolerable."),
    ("guia-ideales",
     "¿Qué son unas condiciones ideales para manipular una carga?",
     "guia_tecnica_mmc_2024",
     "la carga esté próxima al cuerpo, a la altura de las caderas",
     "Carga cerca del cuerpo, a la altura de las caderas, espalda recta, "
     "sin giros, buen agarre."),
    ("guia-distancia",
     "¿A qué distancia del cuerpo hay que llevar la carga?",
     "guia_tecnica_mmc_2024",
     "La distancia recomendada debería ser menor o igual a 25 cm.",
     "A 25 cm o menos."),
    ("guia-alturas",
     "¿Entre qué alturas es aceptable mover una carga?",
     "guia_tecnica_mmc_2024",
     "nunca se deberían elevar cargas por encima de la cabeza",
     "Entre la altura de los nudillos y la de los hombros; nunca por encima "
     "de la cabeza."),
    ("guia-frecuencia",
     "¿Cuántos levantamientos por minuto se consideran un riesgo alto?",
     "guia_tecnica_mmc_2024",
     "el riesgo es alto si se superan alguna de las frecuencias de "
     "manipulación",
     "Más de 15, 12 o 10 por minuto según la duración de la tarea."),
    ("guia-transporte",
     "¿Cuántos metros se puede transportar una carga a mano?",
     "guia_tecnica_mmc_2024",
     "Lo ideal es no transportar la carga una distancia superior a 1 m.",
     "Lo ideal, no más de 1 m; más de 10 m supone grandes demandas."),
    ("guia-empuje",
     "¿Qué fuerza máxima se puede hacer al empujar un carro?",
     "guia_tecnica_mmc_2024",
     "la fuerza aplicada no debe sobrepasar en ningún caso los siguientes "
     "valores",
     "Mujeres 240 N inicial y 150 N mantenida; hombres 360 N y 250 N."),
    ("guia-niosh",
     "¿Qué método recomienda la guía para evaluar levantamientos?",
     "guia_tecnica_mmc_2024",
     "Permite analizar tareas simples, múltiples y compuestas de "
     "levantamiento de cargas.",
     "La ecuación de NIOSH, entre otros."),
    ("guia-momento",
     "¿En qué momento de un levantamiento es mayor el riesgo?",
     "guia_tecnica_mmc_2024",
     "El arranque del levantamiento y el depósito de las cargas",
     "Al arrancar el levantamiento y al depositar la carga."),
    ("guia-agarre",
     "¿Qué profundidad máxima debería tener el punto de agarre de una caja?",
     "guia_tecnica_mmc_2024",
     "es recomendable que la profundidad del punto del agarre no supere los "
     "25 cm",
     "No más de 25 cm."),
    ("ley-derecho",
     "¿Tienen los trabajadores derecho a que les protejan de los riesgos "
     "del trabajo?",
     "ley_31_1995_prevencion_riesgos",
     "Los trabajadores tienen derecho a una protección eficaz en materia de "
     "seguridad y salud en el trabajo.",
     "Sí, a una protección eficaz, con el correlativo deber del "
     "empresario."),
    ("ley-formacion",
     "¿Cuándo tiene que formar la empresa a un trabajador en prevención?",
     "ley_31_1995_prevencion_riesgos",
     "el empresario deberá garantizar que cada trabajador reciba una "
     "formación teórica y práctica",
     "Al contratarlo y cuando cambian sus funciones o se introducen nuevas "
     "tecnologías o equipos."),
    ("ley-sensibles",
     "¿Qué protección especial tienen los trabajadores con discapacidad?",
     "ley_31_1995_prevencion_riesgos",
     "El empresario garantizará de manera específica la protección de los "
     "trabajadores que, por sus propias características personales",
     "Una protección específica como trabajadores especialmente "
     "sensibles."),
    ("ley-trabajador",
     "¿Qué obligaciones tiene el propio trabajador en prevención?",
     "ley_31_1995_prevencion_riesgos",
     "Corresponde a cada trabajador velar, según sus posibilidades",
     "Velar por su seguridad y la de los demás cumpliendo las medidas de "
     "prevención."),
    ("ley-inminente",
     "¿Qué debe hacer la empresa ante un riesgo grave e inminente?",
     "ley_31_1995_prevencion_riesgos",
     "Informar lo antes posible a todos los trabajadores afectados acerca "
     "de la existencia de dicho riesgo",
     "Informar, dar instrucciones para interrumpir la actividad y "
     "abandonar el lugar si es necesario."),
    ("reba-tronco",
     "¿Qué puntuación da REBA a un tronco muy inclinado hacia delante?",
     "ntp_601",
     "20°-60° flexión",
     "Más de 60° de flexión puntúa 4; de 20° a 60°, 3."),
    ("reba-actividad",
     "¿Cuándo se suma un punto por actividad en REBA?",
     "ntp_601",
     "Una o más partes del cuerpo permanecen estáticas",
     "Postura estática más de 1 minuto, repeticiones de más de 4 por "
     "minuto o cambios posturales bruscos."),
    ("reba-niveles",
     "¿Qué hay que hacer si REBA da una puntuación de 11?",
     "ntp_601",
     "Actuación inmediata",
     "Riesgo muy alto: actuación inmediata."),
    ("reba-origen",
     "¿Quién desarrolló el método REBA y para qué sirve?",
     "ntp_601",
     "desarrollado por Hignett y McAtamney (Nottingham, 2000)",
     "Hignett y McAtamney, para estimar el riesgo de desórdenes corporales "
     "relacionados con el trabajo."),
    ("niosh-constante",
     "¿Cuál es la constante de carga de la ecuación de NIOSH?",
     "ntp_477",
     "quedó fijado en 23 kg",
     "23 kg."),
    ("niosh-il3",
     "¿Qué ocurre si el índice de levantamiento supera 3?",
     "ntp_477",
     "Este tipo de tarea es inaceptable desde el punto de vista ergonómico "
     "y debe ser modificada.",
     "La tarea es inaceptable y debe modificarse."),
    ("niosh-frecuencia",
     "¿Durante cuánto tiempo se mide la frecuencia de levantamiento?",
     "ntp_477",
     "en un período de 15 minutos",
     "Como media en un período de 15 minutos."),
    ("niosh-giro",
     "¿Dónde se mide el ángulo de giro en la ecuación de NIOSH?",
     "ntp_477",
     "El ángulo de giro (A) deberá medirse en el origen del movimiento",
     "En el origen, y en el destino si hay que controlar la carga."),
    ("niosh-factores",
     "¿Qué factores multiplican la constante de carga en NIOSH?",
     "ntp_477",
     "HM : factor de distancia horizontal",
     "Distancia horizontal, altura, desplazamiento vertical, asimetría, "
     "frecuencia y agarre."),
]


# Held out: written before measuring them, on other passages; only to
# check the changes made by looking at QUESTIONS.
HOLDOUT = [
    ("h-rd-suelo",
     "¿Influye el estado del suelo en el riesgo de manipular cargas?",
     "rd_487_1997_manipulacion_cargas",
     "Cuando el suelo es irregular y, por tanto, puede dar lugar a "
     "tropiezos",
     "Sí: un suelo irregular o resbaladizo aumenta el riesgo."),
    ("h-rd-espacio",
     "¿Qué pasa si no hay altura suficiente para manipular la carga?",
     "rd_487_1997_manipulacion_cargas",
     "Cuando el espacio libre, especialmente vertical, resulta insuficiente",
     "El espacio libre insuficiente aumenta el riesgo dorsolumbar."),
    ("h-rd-descanso",
     "¿Es un riesgo no tener tiempo para recuperarse entre esfuerzos?",
     "rd_487_1997_manipulacion_cargas",
     "Período insuficiente de reposo fisiológico o de recuperación.",
     "Sí, es una exigencia de la actividad que entraña riesgo."),
    ("h-rd-individuales",
     "¿Qué características de la persona aumentan el riesgo al cargar "
     "pesos?",
     "rd_487_1997_manipulacion_cargas",
     "La falta de aptitud física para realizar las tareas en cuestión.",
     "La falta de aptitud física, ropa inadecuada, falta de formación..."),
    ("h-rd-ambiente",
     "¿El calor o la mala iluminación afectan al riesgo de manipular "
     "cargas?",
     "rd_487_1997_manipulacion_cargas",
     "Cuando la temperatura, humedad o circulación del aire son "
     "inadecuadas.",
     "Sí, son características del medio de trabajo que aumentan el "
     "riesgo."),
    ("h-ley-principios",
     "¿Cuáles son los principios generales de la acción preventiva?",
     "ley_31_1995_prevencion_riesgos",
     "El empresario aplicará las medidas que integran el deber general de "
     "prevención",
     "Evitar los riesgos, evaluar los que no se puedan evitar, combatirlos "
     "en su origen..."),
    ("h-ley-equipos",
     "¿Debe la empresa dar equipos de protección a los trabajadores?",
     "ley_31_1995_prevencion_riesgos",
     "El empresario adoptará las medidas necesarias con el fin de que los "
     "equipos de trabajo sean adecuados",
     "Sí, equipos adecuados y de protección individual cuando haga falta."),
    ("h-ley-emergencia",
     "¿Qué tiene que preparar la empresa para emergencias?",
     "ley_31_1995_prevencion_riesgos",
     "deberá analizar las posibles situaciones de emergencia",
     "Analizar las emergencias y adoptar medidas de primeros auxilios, "
     "incendios y evacuación."),
    ("h-ley-reconocimiento",
     "¿Es obligatorio hacerse los reconocimientos médicos de la empresa?",
     "ley_31_1995_prevencion_riesgos",
     "Esta vigilancia sólo podrá llevarse a cabo",
     "Solo con el consentimiento del trabajador, salvo excepciones."),
    ("h-ley-delegados",
     "¿Quiénes son los delegados de prevención?",
     "ley_31_1995_prevencion_riesgos",
     "Los Delegados de Prevención son los representantes de los "
     "trabajadores con funciones específicas",
     "Representantes de los trabajadores con funciones en prevención."),
    ("h-ley-informacion",
     "¿Qué información debe recibir un trabajador sobre los riesgos de su "
     "puesto?",
     "ley_31_1995_prevencion_riesgos",
     "el empresario adoptará las medidas adecuadas para que los "
     "trabajadores reciban",
     "Los riesgos, las medidas de protección y las de emergencia."),
    ("h-guia-pausas",
     "¿Cómo se puede reducir la fatiga en tareas de carga?",
     "guia_tecnica_mmc_2024",
     "El objetivo principal es permitir que el organismo se recupere antes "
     "de que sobrevenga la aparición de fatiga física.",
     "Con pausas, descansos y cambios de tarea."),
    ("h-guia-pacientes",
     "¿Cómo se debe mover a un paciente que no puede moverse solo?",
     "guia_tecnica_mmc_2024",
     "se priorizará la manipulación entre varias personas",
     "Con ayudas mecánicas y, si no, entre varias personas."),
    ("h-reba-antebrazo",
     "¿Cómo se puntúa el codo en el método REBA?",
     "ntp_601",
     "60°-100° flexión",
     "60-100° de flexión puntúa 1; menos de 60 o más de 100, 2."),
    ("h-reba-agarre",
     "¿Cómo influye la calidad del agarre en la puntuación REBA?",
     "ntp_601",
     "Agarre posible pero no aceptable",
     "Suma de 0 (bueno) a 3 (inaceptable)."),
    ("h-reba-ejemplo",
     "¿Hay algún ejemplo de REBA aplicado en un hospital?",
     "ntp_601",
     "trabajador sanitario para deslizar a un paciente hasta la cabecera "
     "de la cama",
     "Un sanitario que desliza a un paciente hacia la cabecera de la "
     "cama."),
    ("h-niosh-compresion",
     "¿Qué compresión en la espalda se toma como límite en NIOSH?",
     "ntp_477",
     "3,4 kN como fuerza límite de compresión para la aparición de riesgo "
     "de lumbalgia.",
     "3,4 kN."),
    ("h-niosh-limites",
     "¿En qué casos no se puede usar la ecuación de NIOSH?",
     "ntp_477",
     "una sola mano, sentado o arrodillado o cuando se trate de cargar "
     "personas",
     "Con una mano, sentado o arrodillado, cargando personas, de forma "
     "brusca..."),
]


def source(text: str, quote: str) -> tuple[int, int]:
    """The paragraph of the quote, at least ``MIN_SOURCE`` characters."""
    at = text.index(quote)
    start = text.rfind("\n\n", 0, at)
    start = 0 if start < 0 else start + 2
    end = text.find("\n\n", at + len(quote))
    end = len(text) if end < 0 else end
    missing = MIN_SOURCE - (end - start)
    if missing > 0:
        start = max(0, start - missing // 2)
        end = min(len(text), end + missing - missing // 2)
    return start, end


def main() -> int:
    """Write every dataset; stop if a quote is not in the corpus."""
    for name, questions in (("dataset_prl", QUESTIONS),
                            ("dataset_prl_holdout", HOLDOUT)):
        if write(name, questions):
            return 1
    return 0


def write(name: str, questions: list[tuple[str, str, str, str, str]]
          ) -> bool:
    """Write one dataset; True if a quote is not in the corpus."""
    answered, unanswered, problems = [], [], []
    texts = {p: p.read_text(encoding="utf-8")
             for p in sorted(CORPUS.glob("*.md"))}
    for qid, question, stem, quote, answer in questions:
        path = CORPUS / f"{stem}.md"
        if quote not in texts.get(path, ""):
            problems.append(f"{qid}: quote not found in {path}")
            continue
        sources = []
        # The named file first, then any other that says the same.
        for other in [path] + [p for p in texts if p != path]:
            count = texts[other].count(quote)
            if not count:
                continue
            if count > 1:
                print(f"{qid}: quote found {count} times in {other}, the "
                      "first is used", file=sys.stderr)
            start, end = source(texts[other], quote)
            sources.append({"file_path": str(other),
                            "first_character_index": start,
                            "last_character_index": end})
        answered.append({"question_id": qid, "question": question,
                         "answer": answer, "sources": sources})
        unanswered.append({"question_id": qid, "question": question})
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return True
    for folder, items in (("AnsweredQuestions", answered),
                          ("UnansweredQuestions", unanswered)):
        out = OUT / folder / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rag_questions": items},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"{out}: {len(items)} questions")
    return False


if __name__ == "__main__":
    sys.exit(main())
